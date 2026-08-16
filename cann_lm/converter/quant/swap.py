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

import os
from pathlib import Path

import torch
import torch.nn as nn

from .materialize import materialize_int4_with_scale, materialize_int8_per_channel
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
    (per-channel [vocab]), materialize_int8_per_channel → (int8 [vocab,hidden], scale [vocab])。
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
        int8, scale = materialize_int8_per_channel(sd[wkey], sd[skey])
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


def swap_to_quant_linear_from_dopt(
    model: nn.Module,
    dopt_path: str | Path,
    skip: tuple[str, ...] = DEFAULT_SKIP,
    act_quant: bool = True,
) -> dict:
    """Port a dopt GPTQ output -> materialize int4 -> swap QuantLinear (流程 C+D).

    For each nn.Linear (name not matching ``skip``): read the GPTQ-corrected fp
    weight ``{name}.weight`` and the flattened group_scale
    ``{name}.quant_op.weight_quantizer.s`` from the dopt output, **自动推断
    group_size** (从 s_flat numel 反推: groups = numel//out, group = in//groups),
    reshape scale to ``[out, groups]``, materialize to int4
    (``materialize_int4_with_scale``), and swap in a QuantLinear.
    embed_tokens / lm_head are skipped. act_quant 固定 True (W4A8, 当前仅支持).

    Returns a quant-info dict (scheme/group_size/scope/counts) for model_config.json.
    """
    sd = load_dopt_state_dict(dopt_path)
    n_q = 0
    n_skip = 0
    bit = uq = None
    group_size = None
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
        # 自动推断 group_size: s_flat 扁平 [out*groups], groups = numel//out
        groups = s_flat.numel() // out_f
        group_size = in_f // groups
        if in_f % group_size != 0:
            raise ValueError(
                f"{name}: in_features ({in_f}) not divisible by inferred group_size ({group_size})"
            )
        scale2d = s_flat.reshape(out_f, groups)
        bit, uq, offset = _read_dopt_quant_meta(sd, name, out_f, in_f, group_size)
        if uq != 0.0:
            raise ValueError(
                f"{name}: unsigned_quant={uq} (unsigned int4 not supported; "
                "only symmetric uq=0). Re-run dopt with a signed-symmetric strategy."
            )
        if offset is not None and offset.abs().max().item() != 0.0:
            raise NotImplementedError(
                f"{name}: non-zero offset (asymmetric) not supported by 方案 A "
                "quant_matmul_s8s4 (symmetric offset=0 only). Re-run dopt with a "
                "symmetric strategy."
            )
        # 方案 A: nibble-packed int4 [n, k//2] int8 for the quant_matmul_s8s4 op x2 input.
        weight_q, weight_scale = materialize_int4_with_scale(
            w_gptq, scale2d, group_size, signed=True, nbits=bit, packed=True
        )
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
        "act_quant": act_quant,
        "scope": f"all nn.Linear except {list(skip)}",
        "n_quantized": n_q,
        "n_skipped_no_dopt_key": n_skip,
        "dopt_path": str(dopt_path),
        "weight_int4_encoding": "nibble-packed int8 [n,k//2] (low nibble=even k, signed [-8,7]) — 方案 A quant_matmul_s8s4 x2",
        "op": "cann_lm::quant_matmul_s8s4",
        "dopt_bit": bit,
        "dopt_unsigned_quant": uq,
        "offset_supported": "symmetric only (offset=0); non-zero offset rejected",
    }
