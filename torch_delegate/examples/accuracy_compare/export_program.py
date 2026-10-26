# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Command-line export entrypoint for the business export subsystem."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn
import ops  # noqa: F401
from util import parse_args
from util import select_device

REQUEST_NUM = 32
BEAM_WIDTH = 4
BATCH_SIZE = REQUEST_NUM * BEAM_WIDTH
SEQ_LEN = 2
HIDDEN_DIM = 128
NUM_HEADS = 4
VOCAB_SIZE = 32000


class ExportSampleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.logit_proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        self.scale_value = 1.0 / math.sqrt(HIDDEN_DIM // NUM_HEADS)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sin_cache: torch.Tensor,
        cos_cache: torch.Tensor,
        attn_mask: torch.Tensor,
        prefix_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.ops.xllm_placeholder.rope_in_place(
            hidden_states,
            sin_cache,
            cos_cache,
        )
        rope_output = hidden_states
        query = self.q_proj(rope_output)
        key = self.k_proj(rope_output)
        value = self.v_proj(rope_output)
        attn_output, _ = torch.ops.xllm_placeholder.fused_infer_attention_score(
            query,
            key,
            value,
            atten_mask=attn_mask,
            num_heads=NUM_HEADS,
            scale=self.scale_value,
            input_layout="BSH",
        )
        logits = self.logit_proj(attn_output[:, -1, :])
        logits_f32 = logits.to(torch.float32)
        top_k = BEAM_WIDTH
        top_probs_f32, top_indices = torch.topk(logits_f32, k=top_k, dim=-1)
        top_tokens = top_indices.to(torch.int32)
        top_probs = top_probs_f32.to(torch.bfloat16)
        top_probs = top_probs + prefix_scores.unsqueeze(-1).to(torch.bfloat16)
        log_probs = hidden_states.new_zeros((BATCH_SIZE, 1))
        return torch.ops.xllm_placeholder.beam_search(
            log_probs,
            top_tokens,
            top_probs,
        )


def build_sample_inputs(device: torch.device | str) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    device = torch.device(device)
    hidden_states = torch.randn(
        BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=torch.bfloat16
    )
    sin_cache = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=torch.bfloat16)
    cos_cache = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=torch.bfloat16)
    causal_mask = torch.triu(
        torch.full((SEQ_LEN, SEQ_LEN), 1, dtype=torch.uint8, device=device),
        diagonal=1,
    )
    attn_mask = (
        causal_mask.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN)
    )
    prefix_scores = torch.zeros(BATCH_SIZE, device=device)
    return hidden_states, sin_cache, cos_cache, attn_mask, prefix_scores


def _remove_assert_tensor_metadata(ep: torch.export.ExportedProgram) -> int:
    """Remove aten._assert_tensor_metadata nodes from the exported graph.

    torch.export 自动插入的 metadata 校验节点（schema: -> ()）没有计算语义，
    但会被下游转换器（torchair / GE）拒绝。在导出阶段提前移除，
    保证 epair 和 reference 路径均不会再遇到该算子。
    """
    graph = ep.graph
    target = torch.ops.aten._assert_tensor_metadata.default
    removed = 0
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target is target:
            graph.erase_node(node)
            removed += 1
    if removed > 0:
        graph.lint()
        ep.graph_module.recompile()
        print(f"[Strip] 移除了 {removed} 个 _assert_tensor_metadata 节点")
    return removed


def export_sample_program(
    output_path: Path,
    device_name: str = "auto",
    print_graph: bool = False,
) -> tuple[Path, torch.export.ExportedProgram]:
    device = select_device(device_name)
    model = ExportSampleModel().eval().to(device)
    model = model.to(torch.bfloat16)
    sample_inputs = build_sample_inputs(device)
    exported_program = torch.export.export(model, sample_inputs, strict=False)

    _remove_assert_tensor_metadata(exported_program)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, output_path)

    graph_path = output_path.with_suffix(".graph.py")
    print(exported_program)
    graph_path.write_text(str(exported_program.graph_module.graph), encoding="utf-8")

    return output_path, exported_program


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    exported_path, _ = export_sample_program(
        output_path=output_path,
        device_name=args.device,
        print_graph=args.print_graph,
    )
    print(f"Exported program saved to: {exported_path}")
    print(f"Graph text saved to: {exported_path.with_suffix('.graph.py')}")


if __name__ == "__main__":
    main()
