"""Export a dynamic-shape epair for qwen3-0.6B (seq_len + batch symbolic).

Unlike the static ExportWrapper (which passes past_key_values=None / embeddings=None
as None placeholders that torchair rejects), this uses a minimal prefill wrapper that
takes only tensor inputs (input_ids, attention_mask, position_ids) and returns logits.
torch.export with dynamic_shapes makes seq_len + batch symbolic → one graph covers any
prefill length. No KV cache (prefill-only path; past_key_values defaults None inside
the model → no-cache branch, no None placeholder lifted).

Usage:
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python export_dynamic_shape.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn
from torch.export import Dim

from cann_lm.models.qwen3 import Qwen3ForCausalLM, Qwen3TextConfig, load_qwen3_weights
from cann_lm.converter.export.epair_utils import consolidated_weights_save

HF_PATH = "/mnt/workspace/data/models/qwen3-0.6B"
OUT_DIR = Path(__file__).resolve().parent / "generate" / "qwen3_dynamic"
SAMPLE_SEQ_LEN = 8  # concrete sample for tracing; dims marked dynamic below


class Qwen3PrefillDynWrapper(nn.Module):
    """Minimal prefill wrapper: only tensor inputs, returns logits only.

    No past_key_values / embeddings args → no None placeholders → torchair OK.
    past_key_values defaults None inside Qwen3ForCausalLM → no-cache prefill branch.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return out["logits"]


def _reorder_pack_attrs(epair) -> int:
    """Reorder Pack op attrs to [axis, N, ...] to match the GE IR definition.

    torchair emits Pack with attrs in [N, axis] order; the GE Pack IR def
    (ge/.../op_reg.h) declares [axis, N] (axis before N, N REQUIRED_ATTR).
    The mismatch trips GE PrepareDynShape -> RecoverIrAttrNames
    "attribute order has changed". Reorder each Pack node's attr dict so axis
    comes before N (matching the IR def).
    """
    graph = epair.ep_json.get("graph_module", {}).get("graph", {})
    fixed = 0
    for n in graph.get("nodes", []):
        ai = n.get("ascend_ir")
        if not isinstance(ai, dict):
            continue
        for o in ai.get("ops", []):
            if not isinstance(o, dict) or o.get("type") != "Pack":
                continue
            attr = o.get("attr")
            if not isinstance(attr, dict) or "axis" not in attr or "N" not in attr:
                continue
            new = {"axis": attr["axis"], "N": attr["N"]}
            for k, v in attr.items():
                if k not in ("axis", "N"):
                    new[k] = v
            o["attr"] = new
            fixed += 1
    return fixed


def main() -> None:
    cfg = Qwen3TextConfig.from_pretrained(HF_PATH)
    model = Qwen3ForCausalLM(cfg).to(torch.float16).eval()
    model = load_qwen3_weights(model, HF_PATH, device="cpu", dtype=torch.float16)
    wrapper = Qwen3PrefillDynWrapper(model).eval()

    # concrete sample (for tracing) — dims below marked symbolic
    b, s = 1, SAMPLE_SEQ_LEN
    input_ids = torch.randint(0, cfg.vocab_size, (b, s), dtype=torch.int64)
    min_val = torch.finfo(torch.float16).min
    causal = torch.triu(torch.full((s, s), min_val, dtype=torch.float16), diagonal=1)
    attention_mask = causal.unsqueeze(0).unsqueeze(0).expand(b, 1, s, s).contiguous()
    position_ids = torch.arange(s, dtype=torch.int64).unsqueeze(0)
    sample_inputs = (input_ids, attention_mask, position_ids)

    # dynamic shapes: batch 固定 1（模型代码特化 batch），只 seq_len 符号化
    seq_dim = Dim("seq_len", min=1, max=cfg.max_position_embeddings)
    dynamic_shapes = {
        "input_ids": {1: seq_dim},
        "attention_mask": {2: seq_dim, 3: seq_dim},
        "position_ids": {1: seq_dim},
    }

    print(f"[1/3] torch.export (dynamic batch+seq_len, sample s={s})")
    ep = torch.export.export(wrapper, sample_inputs, dynamic_shapes=dynamic_shapes, strict=False)
    print(f"      export OK; graph nodes: {len(list(ep.graph_module.graph.nodes))}")

    import torch_delegate
    print(f"[2/3] export_to_epair")
    epair = torch_delegate.export_to_epair(ep)

    # torchair 导出 Pack 时属性顺序为 [N, axis]，GE IR 定义是 [axis, N]，不一致会导致
    # C++ 编译 PrepareDynShape 阶段 RecoverIrAttrNames "attribute order has changed" 失败。
    # 落盘前把 Pack 节点属性重排成 [axis, N, ...]。
    _fixed = _reorder_pack_attrs(epair)
    if _fixed:
        print(f"      reordered Pack attrs (axis before N) in {_fixed} nodes")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    consolidated_weights_save(epair, OUT_DIR, save_weights=True, max_file_size=512 * 1024 * 1024)
    print(f"[3/3] saved dynamic-shape epair → {OUT_DIR}")

    # sanity: re-load the ep_json, check a node has symbolic/dynamic shape
    import json
    ep_json = json.load(open(OUT_DIR / "ep_json_with_ascend_ir.json"))
    g = ep_json["graph_module"]["graph"]
    print(f"      ep_json nodes: {len(g['nodes'])}; inputs: {len(g.get('inputs', []))}")


if __name__ == "__main__":
    main()
