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

"""Swap nn.Linear modules for QuantLinear (pre-export op IR replacement)."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from .materialize import (
    materialize_int4_per_group_symmetric,
    materialize_int8_per_channel_symmetric,
)
from .quant_linear import QuantLinear

# Linear swap skips names matching these substrings. embed_tokens* are nn.Embedding
# (consumed by swap_embeddings_from_dopt for W8 host gather). lm_head is NOT skipped
# -> quantized to W4; host-embed + NPU-lm_head forces tie_word_embeddings untie.
DEFAULT_SKIP = ("embed_tokens", "embed_tokens_per_layer")


def swap_embeddings_from_dopt(
    sd: dict, model: nn.Module
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Port dopt Quant_Embed_MinMax embed (int8 + per-channel scale) for host lookup.

    Reads {name}.weight (fakequant) + {name}.quant_op.weight_quantizer.s per
    nn.Embedding, materializes to (int8 [vocab,hidden], scale [vocab]). NOT swapped
    into a graph module — embeddings are externalized for host gather (unlike
    Linear->QuantLinear). Skips entries without a scale (e.g. unquantized
    gemma4 embed_tokens_per_layer). Returns {wrapper_fqn: (int8, scale)}.
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
    else:
        # .pth (torch.save state_dict)
        sd = torch.load(str(p), map_location="cpu")
    # Normalize FQN: transformers' Qwen3_5ForCausalLM.save_pretrained wraps the text
    # llm state_dict into VLM structure (model.language_model.*) when saving; gemma4
    # saves as-is (model.*). Strip language_model. so the dopt sd is text-only (model.*)
    # for both, matching cann_lm's text-only model.* layout.
    if any(k.startswith("model.language_model.") for k in sd):
        sd = {
            ("model." + k[len("model.language_model."):])
            if k.startswith("model.language_model.") else k: v
            for k, v in sd.items()
        }
    return sd


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


# dopt strategy defaults (aligned with dopt_lm/quantizer/strategy.py strategy_dict weight fields).
# converter venv doesn't install dopt_lm, so hardcoded; update here when adding a weight-quant strategy.
# Only strategy config (group_size/bit/blocksize/per_channel/weight_algo) — unsigned_quant/offset
# are GPTQ runtime state (dopt writes them to safetensors buffers, not in config).
# dopt 53ad5ec changed Quant_INT8-4_ECO group_size from 256 to 128; synced here.
_STRATEGY_DEFAULT_PARAMS: dict[str, dict] = {
    "Quant_INT8-4_ECO": {
        "group_size": 128, "bit": 4, "blocksize": 128,
        "per_channel": True, "weight_algo": "omni_group_min_max",
    },
}


def _read_quant_param_map(dopt_path: str | Path | None) -> dict[str, dict]:
    """Parse {layer_name: weight-side quant params} from dopt_config.json in the dopt output dir.

    Parsing (aligned with dopt strategy.get_optimize_cfg):
    1. quant_strategy suffix (e.g. "Quant_INT8-4_ECO;128") overrides group_size
    2. _STRATEGY_DEFAULT_PARAMS provides defaults (group_size/bit/blocksize/per_channel/weight_algo)

    Returns {name: {group_size, bit, blocksize, per_channel, weight_algo}};
    absent dopt_config.json / unconfigured layer / strategy without defaults -> not in dict
    (caller falls back to _infer_group_size + _read_dopt_quant_meta from safetensors buffers).
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
        # Suffix parse (e.g. "Quant_INT8-4_ECO;128" -> strategy="Quant_INT8-4_ECO", gs=128).
        gs_override = None
        if ";" in strategy:
            strategy, gs_str = strategy.split(";", 1)
            try:
                gs_override = int(gs_str)
            except ValueError:
                continue
        defaults = _STRATEGY_DEFAULT_PARAMS.get(strategy)
        if defaults is None:
            continue  # strategy has no weight defaults (e.g. Quant_Embed_MinMax is embedding-side)
        params = dict(defaults)
        if gs_override is not None:
            params["group_size"] = gs_override
        out[name] = params
    return out


def _infer_group_size(
    s_flat: torch.Tensor, out_f: int, in_f: int, name: str
) -> tuple[int, int]:
    """Infer (group_size, groups) from the flattened dopt scale s_flat [out*groups].

    Supports pergroup (numel = out * in/group) and perchannel (numel = out,
    group_size = in_f, groups = 1). Rejects pertensor (numel == 1), non-1D shapes,
    numel not divisible by out_f, and in_f not divisible by the inferred group_size.
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
    """Port a loaded dopt GPTQ sd -> materialize int4 -> swap QuantLinear in place.

    Takes an already-loaded sd (from load_dopt_state_dict) to avoid re-reading.
    lm_head is not in ``skip`` -> also quantized.

    Quant params source (priority):
    1. param_map[name] (parsed from dopt_config.json: strategy suffix + _STRATEGY_DEFAULT_PARAMS)
       -> group_size/bit/blocksize/per_channel/weight_algo
    2. fallback:
       - group_size: _infer_group_size (from s numel)
       - bit/unsigned_quant/offset: _read_dopt_quant_meta (from safetensors buffers;
         these are GPTQ runtime state, not stored in dopt_config.json)

    Returns a quant-info dict (scheme/group_size/bit/... for model_config.json).
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
            # from dopt_config.json: group_size/bit/blocksize/per_channel/weight_algo
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
                    f"inconsistent with s shape"
                )
            bit_cfg = params.get("bit")
            blocksize = params.get("blocksize")
            per_channel = params.get("per_channel")
            weight_algo = params.get("weight_algo")
        else:
            # fallback: infer group_size from s numel (older output without dopt_config.json, or unconfigured layer)
            group_size, groups = _infer_group_size(s_flat, out_f, in_f, name)
            bit_cfg = None  # bit also falls back to safetensors buffer
        scale2d = s_flat.reshape(out_f, groups)
        # bit/unsigned_quant/offset still read from safetensors buffers (GPTQ runtime state, not in config)
        # config bit takes priority if present, else the safetensors buffer
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
        # Pre-transpose weight_q [n,k//2]->[k//2,n] and scale [n,k//g]->[k//g,n]: npu::npu_quant_matmul
        # meta takes N = x2.size(-1); _quant_matmul_converter hardcodes transpose_x2=False.
        weight_q, weight_scale = materialize_int4_per_group_symmetric(
            w_gptq, scale2d, group_size, signed=True, nbits=bit, packed=True
        )
        weight_q = weight_q.transpose(-1, -2).contiguous()
        weight_scale = weight_scale.transpose(-1, -2).contiguous()
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
    """Backward-compatible thin wrapper: load_dopt_state_dict + _read_quant_param_map
    + _swap_linear_from_sd. Use for offline scene-1 / standalone Linear swap; the
    main process (scenes 1/2) uses swap_from_dopt (one sd load for both Linear swap
    + embed extraction).
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
    """One dopt sd load -> Linear swap + Embedding extraction.

    Shared by the main process for scene 1 (--quant-path) and scene 2
    (--quant-method gptq); avoids the double load IO of
    swap_to_quant_linear_from_dopt's internal load + swap_embeddings_from_dopt's
    external load.

    Returns:
        (quant_info, embed_quant):
            quant_info: quant_info field for model_config.json
            embed_quant: {wrapper_fqn: (int8, scale)} for saver host lookup (may be empty)
    """
    sd = load_dopt_state_dict(dopt_path)
    param_map = _read_quant_param_map(dopt_path)
    info = _swap_linear_from_sd(
        model, sd, skip=skip, act_quant=act_quant, param_map=param_map,
    )
    info["dopt_path"] = str(dopt_path)
    embed_quant = swap_embeddings_from_dopt(sd, model)
    return info, embed_quant
