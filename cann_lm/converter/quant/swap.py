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

from .materialize import materialize_int4, materialize_int4_with_scale
from .quant_linear import QuantLinear

# names containing any of these substrings are NOT quantized
DEFAULT_SKIP = ("embed_tokens", "lm_head", "embed_tokens_per_layer")


def _resolve_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    """Return (parent module, leaf attr name) for a dotted module name."""
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def swap_to_quant_linear(
    model: nn.Module,
    group_size: int = 128,
    skip: tuple[str, ...] = DEFAULT_SKIP,
    act_quant: bool = True,
) -> nn.Module:
    """Replace target nn.Linear modules with QuantLinear (materialize int4 weight).

    Walks ``model.named_modules()``; for each nn.Linear whose name does not
    contain a ``skip`` substring, materializes its weight to int4 + per-group
    scale and swaps in a QuantLinear. embed_tokens / lm_head are skipped by
    default. ``act_quant=False`` produces W4A16 (weight-only, no act quant).

    Returns the same model (mutated in place).
    """
    for name, m in list(model.named_modules()):
        if not isinstance(m, nn.Linear):
            continue
        if any(s in name for s in skip):
            continue
        weight_q, weight_scale = materialize_int4(m.weight, group_size)
        ql = QuantLinear(weight_q, weight_scale, group_size, bias=m.bias, act_quant=act_quant)
        parent, child = _resolve_parent(model, name)
        setattr(parent, child, ql)
    return model


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


def swap_to_quant_linear_from_dopt(
    model: nn.Module,
    dopt_path: str | Path,
    group_size: int = 256,
    skip: tuple[str, ...] = DEFAULT_SKIP,
    act_quant: bool = True,
) -> dict:
    """Port a dopt GPTQ output -> materialize int4 -> swap QuantLinear (流程 C+D).

    For each nn.Linear (name not matching ``skip``): read the GPTQ-corrected fp
    weight ``{name}.weight`` and the flattened group_scale
    ``{name}.quant_op.weight_quantizer.s`` from the dopt output, reshape scale to
    ``[out, in//group]``, materialize to int4 (``materialize_int4_with_scale``),
    and swap in a QuantLinear. embed_tokens / lm_head are skipped.

    Returns a quant-info dict (scheme/group_size/scope/counts) for model_config.json.
    """
    sd = load_dopt_state_dict(dopt_path)
    n_q = 0
    n_skip = 0
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
        if in_f % group_size != 0:
            raise ValueError(
                f"{name}: in_features ({in_f}) not divisible by group_size ({group_size})"
            )
        scale2d = s_flat.reshape(out_f, in_f // group_size)
        weight_q, weight_scale = materialize_int4_with_scale(w_gptq, scale2d, group_size)
        ql = QuantLinear(weight_q, weight_scale, group_size, bias=m.bias, act_quant=act_quant)
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
        "weight_int4_encoding": "unpack+signed[-8,7] placeholder (CANN convention TBD)",
    }
