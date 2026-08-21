# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Swap nn.Linear modules for QuantLinear (导出前算子 IR 替换)."""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from .materialize import materialize_int4_per_group_symmetric, materialize_int8_per_channel_symmetric
from .quant_linear import QuantLinear

# names containing any of these substrings are NOT quantized by the Linear swap.
# embed_tokens / embed_tokens_per_layer are nn.Embedding (not nn.Linear) — never
# reached by the Linear swap anyway; they're consumed by swap_embeddings_from_dopt
# (W8 host gather, requires the converter --quant-embeddings flag).
# lm_head: NOT skipped — it's a big NPU-side Linear, quantized to W4 (QuantLinear).
# For host-embed + NPU-lm_head deployment (tie_word_embeddings must be untied:
# embed on host, lm_head on NPU — different compute domains, can't share weight).
DEFAULT_SKIP = ("embed_tokens", "embed_tokens_per_layer")


def swap_embeddings_from_dopt(
    sd: dict, model: nn.Module
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Port dopt Quant_Embed_MinMax embed int8+per-channel scale for host lookup.

    遍历 model 的 nn.Embedding (embed_tokens + gemma4 embed_tokens_per_layer 子类),
    从 dopt sd 读 {name}.weight (fakequant) + {name}.quant_op.weight_quantizer.s
    (per-channel [vocab]), materialize_int8_per_channel_symmetric → (int8 [vocab,hidden], scale [vocab])。
    返回 {wrapper_fqn: (int8, scale)} (wrapper.model = ForCausalLM → 'model.' 前缀)。
    **不** swap 成图模块 — embedding 外置给 host gather (区别于 Linear→QuantLinear)。
    跳过 sd 里无 scale 的 (如 gemma4 embed_tokens_per_layer 未量化时)。
    """
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Embedding):
            continue
        wkey = f"{name}.weight"
        skey = f"{name}.quant_op.weight_quantizer.s"
        if wkey not in sd or skey not in sd:
            continue
        int8, scale = materialize_int8_per_channel_symmetric(sd[wkey], sd[skey])
        out[f"model.{name}.weight"] = (int8, scale)  # wrapper-level FQN
    return out


def _resolve_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Return (parent module, leaf attr name) for a dotted module name."""
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def load_dopt_state_dict(dopt_path: str | Path) -> dict:
    """Load a dopt GPTQ output (QLinear state_dict) as a plain {name: tensor} dict.

    Accepts: a ``.safetensors`` file, a ``.pth`` (torch.save state_dict), or a
    directory containing ``model.safetensors`` (dopt ``--hf_type_save`` output).
    Keys look like ``model.layers.N.self_attn.q_proj.weight`` (GPTQ-corrected fp)
    and ``model.layers.N.self_attn.q_proj.quant_op.weight_quantizer.s`` (group_scale,
    flattened ``[out * in//group]``).
    """
    p = Path(dopt_path)
    if p.is_dir():
        st = p / "model.safetensors"
        if not st.exists():
            raise FileNotFoundError(f"No model.safetensors under {p}")
        p = st
    if not p.exists():
        raise FileNotFoundError(f"dopt output not found: {p}")
    if p.suffix == ".safetensors":
        from safetensors import safe_open
        sd: dict = {}
        with safe_open(str(p), "pt") as sf:
            for k in sf.keys():
                sd[k] = sf.get_tensor(k)
        return sd
    # .pth (torch.save state_dict)
    return torch.load(str(p), map_location="cpu")


def _read_dopt_quant_meta(
    sd: dict, name: str, out_f: int, in_f: int, group_size: int
) -> tuple[int, float, torch.Tensor | None]:
    """Read dopt quantizer buffers (bit/unsigned_quant/offset) for one Linear.

    dopt saves these alongside ``s`` under
    ``{name}.quant_op.weight_quantizer.{bit,unsigned_quant,offset}`` (verified:
    bit=int64 scalar, unsigned_quant=fp32 scalar, offset=fp32 flattened
    [out*groups], same layout as ``s``).

    Returns (bit, unsigned_quant, offset_2d). Falls back to (4, 0.0, None)
    when the buffers are absent (older dopt outputs that only saved weight+s).
    offset is reshaped from [out*groups] to [out, in//group] (same as scale).
    """
    pre = f"{name}.quant_op.weight_quantizer"
    bit_key = f"{pre}.bit"
    uq_key = f"{pre}.unsigned_quant"
    off_key = f"{pre}.offset"
    if bit_key not in sd or uq_key not in sd or off_key not in sd:
        return 4, 0.0, None
    bit = int(sd[bit_key].item())
    uq = float(sd[uq_key].item())
    offset = sd[off_key].float().reshape(out_f, in_f // group_size)
    return bit, uq, offset


# dopt 策略默认参数 (与 dopt_lm/quantizer/strategy.py 的 strategy_dict weight 字段对齐).
# converter venv 不装 dopt_lm 不能 import,硬编码;新增带 weight 量化的策略时需同步更新此处.
# 注意: 只含策略配置 (group_size/bit/blocksize/per_channel/weight_algo);
#       unsigned_quant/offset 是 GPTQ 运行时状态 (dopt 写到 safetensors buffer),不在 config.
_STRATEGY_DEFAULT_PARAMS: dict[str, dict] = {
    "Quant_act_weight_eco": {
        "group_size": 256, "bit": 4, "blocksize": 128,
        "per_channel": True, "weight_algo": "omni_group_min_max",
    },
    "Quant_lm_head": {
        "group_size": 256, "bit": 4, "blocksize": 128,
        "per_channel": True, "weight_algo": "omni_group_min_max",
    },
}


def _read_quant_param_map(dopt_path: str | Path | None) -> dict[str, dict]:
    """从 dopt 产物目录的 dopt_config.json 解析 {层名: weight 侧量化参数 dict}.

    解析逻辑 (与 dopt strategy.get_optimize_cfg 对齐):
    1. quant_strategy 后缀 (如 "Quant_act_weight_eco;128") 覆盖 group_size
    2. _STRATEGY_DEFAULT_PARAMS 提供默认值 (group_size/bit/blocksize/per_channel/weight_algo)

    Returns:
        {name: {group_size, bit, blocksize, per_channel, weight_algo}};
        无 dopt_config.json / 层未配置 / 策略无默认参数 → 不在该 dict
        (调用方 fallback 到 _infer_group_size + _read_dopt_quant_meta 从 safetensors 反推/读 buffer)
    """
    if dopt_path is None:
        return {}
    cfg_path = Path(dopt_path)
    if cfg_path.is_dir():
        cfg_path = cfg_path / "dopt_config.json"
    if not cfg_path.is_file():
        return {}
    with open(cfg_path) as f:
        cfg = json.load(f)
    layer_strategy = cfg.get("layer_strategy", {})

    out: dict[str, dict] = {}
    for name, layer in layer_strategy.items():
        strategy = layer.get("quant_strategy", "")
        # 后缀解析 (如 "Quant_act_weight_eco;128" → strategy="Quant_act_weight_eco", gs=128)
        gs_override = None
        if ";" in strategy:
            strategy, gs_str = strategy.split(";", 1)
            try:
                gs_override = int(gs_str)
            except ValueError:
                continue
        defaults = _STRATEGY_DEFAULT_PARAMS.get(strategy)
        if defaults is None:
            continue  # 策略无默认参数 (如 Quant_Embed_MinMax 是 embedding,不在此处)
        params = dict(defaults)  # 浅拷贝默认值
        if gs_override is not None:
            params["group_size"] = gs_override
        out[name] = params
    return out


def _infer_group_size(
    s_flat: torch.Tensor, out_f: int, in_f: int, name: str
) -> tuple[int, int]:
    """从 dopt scale s_flat (扁平 [out*groups]) 推断 (group_size, groups).

    支持模式:
      - pergroup (numel = out * in/group, group_size = 推断值)
      - perchannel (numel = out, group_size = in_f, groups = 1)

    拒绝:
      - pertensor (numel == 1)
      - s.ndim != 1 或 numel 不是 out_f 整数倍 (非 perchannel/pergroup 形状)
      - in_f 不整除 group_size (奇怪 dopt 策略)
    """
    if s_flat.numel() == 1:
        raise ValueError(
            f"{name}: pertensor scale (numel=1) not supported, use perchannel/pergroup"
        )
    if s_flat.dim() != 1:
        raise ValueError(
            f"{name}: expected 1D flat scale, got {s_flat.dim()}D shape {tuple(s_flat.shape)}"
        )
    if s_flat.numel() % out_f != 0:
        raise ValueError(
            f"{name}: scale numel ({s_flat.numel()}) not divisible by out_f ({out_f})"
        )
    groups = s_flat.numel() // out_f
    if groups == 0:
        raise ValueError(f"{name}: scale numel ({s_flat.numel()}) < out_f ({out_f})")
    group_size = in_f // groups
    if in_f % group_size != 0:
        raise ValueError(
            f"{name}: in_features ({in_f}) not divisible by inferred group_size ({group_size})"
        )
    return group_size, groups


def _swap_linear_from_sd(
    model: nn.Module,
    sd: dict,
    skip: tuple[str, ...] = DEFAULT_SKIP,
    act_quant: bool = True,
    param_map: dict[str, dict] | None = None,
) -> dict:
    """Port a dopt GPTQ sd -> materialize int4 -> swap QuantLinear (核心逻辑).

    接收已加载的 sd (load_dopt_state_dict 返回),避免重复 IO. 遍历 nn.Linear (name 不含 skip):
    读 GPTQ-corrected fp weight + flattened group_scale, materialize int4 (nibble-packed),
    替换为 QuantLinear. lm_head 不在 skip → 也被量化.

    量化参数来源 (优先级):
    1. param_map[name] (从 dopt_config.json 解析,方案 C:策略名后缀 + _STRATEGY_DEFAULT_PARAMS)
       提供 group_size/bit/blocksize/per_channel/weight_algo
    2. fallback:
       - group_size: _infer_group_size (从 s numel 反推,带 4 个断言)
       - bit/unsigned_quant/offset: _read_dopt_quant_meta (从 safetensors buffer 读,
         这三个是 GPTQ 运行时状态,dopt_config.json 不存)

    Returns quant-info dict (scheme/group_size/bit/...for model_config.json).
    """
    n_q = 0
    n_skip = 0
    bit = uq = None
    group_size = None
    blocksize = per_channel = weight_algo = None
    param_map = param_map or {}
    for name, m in list(model.named_modules()):
        if not isinstance(m, nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        wkey = f"{name}.weight"
        skey = f"{name}.quant_op.weight_quantizer.s"
        if wkey not in sd or skey not in sd:
            n_skip += 1
            continue
        w_gptq = sd[wkey].float()
        s_flat = sd[skey].float()
        out_f, in_f = w_gptq.shape
        params = param_map.get(name)
        if params is not None and "group_size" in params:
            # 方案 C: 从 dopt_config.json 拿 group_size/bit/blocksize/per_channel/weight_algo
            group_size = params["group_size"]
            if in_f % group_size != 0:
                raise ValueError(
                    f"{name}: in_features ({in_f}) not divisible by group_size "
                    f"from dopt_config ({group_size})"
                )
            groups = in_f // group_size
            if s_flat.numel() != out_f * groups:
                raise ValueError(
                    f"{name}: scale numel ({s_flat.numel()}) != out_f * groups "
                    f"({out_f} * {groups} = {out_f * groups}); dopt_config group_size={group_size} "
                    f"与 s 形状不一致"
                )
            bit_cfg = params.get("bit")
            blocksize = params.get("blocksize")
            per_channel = params.get("per_channel")
            weight_algo = params.get("weight_algo")
        else:
            # fallback: 从 s numel 反推 group_size (老产物无 dopt_config.json 或层未配置)
            group_size, groups = _infer_group_size(s_flat, out_f, in_f, name)
            bit_cfg = None  # bit 也 fallback 到 safetensors buffer
        scale2d = s_flat.reshape(out_f, groups)
        # bit/unsigned_quant/offset 仍从 safetensors buffer 读 (GPTQ 运行时状态,不在 config)
        # 若 config 给了 bit 则优先用 config 的 (策略配置),否则用 safetensors buffer
        bit_sd, uq, offset = _read_dopt_quant_meta(sd, name, out_f, in_f, group_size)
        bit = bit_cfg if bit_cfg is not None else bit_sd
        if uq != 0.0:
            raise ValueError(
                f"{name}: unsigned_quant={uq} (unsigned int4 not supported; "
                "only symmetric uq=0). Re-run dopt with a signed-symmetric strategy."
            )
        if offset is not None and offset.abs().max().item() != 0.0:
            raise NotImplementedError(
                f"{name}: non-zero offset (asymmetric) not supported by "
                "npu::npu_quant_matmul (symmetric offset=0 only). Re-run dopt with a "
                "symmetric strategy."
            )
        # nibble-packed int4 int8: materialize -> [n, k//2], then pre-transpose to
        # [k//2, n] (npu::npu_quant_matmul meta takes N = x2.size(-1) and ignores
        # transpose_x2; GE converter _quant_matmul_converter hardcodes transpose_x2=False,
        # so x2 must be [k//2, n] for size(-1) = n = out_features). scale同理 [n,k//g] -> [k//g,n].
        weight_q, weight_scale = materialize_int4_per_group_symmetric(
            w_gptq, scale2d, group_size, signed=True, nbits=bit, packed=True
        )
        weight_q = weight_q.transpose(-1, -2).contiguous()          # [n, k//2] -> [k//2, n]
        weight_scale = weight_scale.transpose(-1, -2).contiguous()   # [n, k//g] -> [k//g, n]
        ql = QuantLinear(
            weight_q, weight_scale, group_size, bias=m.bias, act_quant=act_quant,
        )
        parent, child = _resolve_parent(model, name)
        setattr(parent, child, ql)
        n_q += 1
    scheme = "W4A8-s8s4" if act_quant else "W4A16-s4"
    return {
        "scheme": scheme,
        "group_size": group_size,
        "bit": bit,
        "blocksize": blocksize,
        "per_channel": per_channel,
        "weight_algo": weight_algo,
        "act_quant": act_quant,
        "scope": f"all nn.Linear except {list(skip)}",
        "n_quantized": n_q,
        "n_skipped_no_dopt_key": n_skip,
        "weight_int4_encoding": "nibble-packed int8 [k//2,n] pre-transposed (low nibble=even k, signed [-8,7]) — npu::npu_quant_matmul x2",
        "op": "npu::npu_quant_matmul",
        "dopt_unsigned_quant": uq,
        "offset_supported": "symmetric only (offset=0); non-zero offset rejected",
    }


def swap_to_quant_linear_from_dopt(
    model: nn.Module,
    dopt_path: str | Path,
    skip: tuple[str, ...] = DEFAULT_SKIP,
    act_quant: bool = True,
) -> dict:
    """Port a dopt GPTQ output -> materialize int4 -> swap QuantLinear (流程 C+D).

    向后兼容薄 wrapper: load_dopt_state_dict + _read_group_size_map + 调 _swap_linear_from_sd.
    仅供场景 1 离线用户或单独 Linear swap 用;主进程场景 1/2 用 swap_from_dopt
    (一次 load sd 同时做 Linear swap + embed 抽取).
    """
    sd = load_dopt_state_dict(dopt_path)
    param_map = _read_quant_param_map(dopt_path)
    info = _swap_linear_from_sd(
        model, sd, skip=skip, act_quant=act_quant, param_map=param_map,
    )
    info["dopt_path"] = str(dopt_path)
    return info


def swap_from_dopt(
    model: nn.Module,
    dopt_path: str | Path,
    skip: tuple[str, ...] = DEFAULT_SKIP,
    act_quant: bool = True,
) -> tuple[dict, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """组合: 一次 load dopt sd → Linear swap + Embedding 抽取.

    主进程场景 1 (--quant-path) + 场景 2 (--quant-method gptq) 共用此函数, 避免
    swap_to_quant_linear_from_dopt 内部 load + swap_embeddings_from_dopt 外部 load
    的二次 IO (7B safetensors 二次 load 几秒浪费).

    Returns:
        (quant_info, embed_quant):
            quant_info: 写入 model_config.json 的 quant_info 字段
            embed_quant: {wrapper_fqn: (int8, scale)} 给 saver host 查表 (可能为空 dict)
    """
    sd = load_dopt_state_dict(dopt_path)
    param_map = _read_quant_param_map(dopt_path)
    info = _swap_linear_from_sd(
        model, sd, skip=skip, act_quant=act_quant, param_map=param_map,
    )
    info["dopt_path"] = str(dopt_path)
    embed_quant = swap_embeddings_from_dopt(sd, model)
    return info, embed_quant
