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

"""Unified entrypoint: export .pt2 and compile to .epair.

Usage:
  # Step 1: export ExportedProgram
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m cann_lm.models.export_epair_with_customop export \
      --output ./generate/sample.pt2 --device cpu

  # Step 2: compile to epair
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m cann_lm.models.export_epair_with_customop compile \
      --ep-path ./generate/sample.pt2 --epair-path ./generate/sample.epair
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn

from cann_lm.models.fused_ops.ops import *  # noqa: F401  (registers cann_lm fused ops: rms_norm/add_rms_norm/rotary_mul/chunk_gated_delta_rule/quant_matmul)
from cann_lm.converter.export.ge_converters import register_ge_converters

# customop 专用算子 (三模型未用, 从 cann_lm/models/fused_ops/ 移至此):
#   beam_search / fused_infer_attention_score / rope_in_place (schema + fake meta)
import beam_search  # noqa: F401  (side-effect: register schema + fake)
import fused_infer_attention_score  # noqa: F401
import rope_in_place  # noqa: F401


def register_customop_converters() -> None:
    """注册 customop 专用算子的 GE converter (从 ge_converter 移出, 三模型未用)."""
    from typing import Any
    from torchair import register_fx_node_ge_converter
    from torchair.ge import custom_op
    from torchair._ge_concrete_graph import ge_apis as ge

    @register_fx_node_ge_converter(torch.ops.cann_lm.rope_in_place.default)
    def _convert_rope_in_place(
        input: Any,
        sin_cache: Any,
        cos_cache: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return custom_op(
            "RopeInPlace",
            inputs={"input": input, "sin_cache": sin_cache, "cos_cache": cos_cache},
            outputs=["output"],
        )

    @register_fx_node_ge_converter(torch.ops.cann_lm.beam_search.default)
    def _convert_beam_search(
        log_probs: Any,
        top_tokens: Any,
        top_probs: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return custom_op(
            "BeamSearch",
            inputs={
                "log_probs": log_probs,
                "top_tokens": top_tokens,
                "top_probs": top_probs,
            },
            outputs=["out_token_ids", "out_token_index", "out_log_probs"],
        )

    @register_fx_node_ge_converter(torch.ops.cann_lm.fused_infer_attention_score.default)
    def _convert_fused_infer_attention_score(
        query: Any,
        key: Any,
        value: Any,
        *,
        pse_shift: Any = None,
        atten_mask: Any = None,
        actual_seq_lengths: Any = None,
        actual_seq_lengths_kv: Any = None,
        dequant_scale1: Any = None,
        quant_scale1: Any = None,
        dequant_scale2: Any = None,
        quant_scale2: Any = None,
        quant_offset2: Any = None,
        antiquant_scale: Any = None,
        antiquant_offset: Any = None,
        block_table: Any = None,
        query_padding_size: Any = None,
        kv_padding_size: Any = None,
        key_antiquant_scale: Any = None,
        key_antiquant_offset: Any = None,
        value_antiquant_scale: Any = None,
        value_antiquant_offset: Any = None,
        key_shared_prefix: Any = None,
        value_shared_prefix: Any = None,
        actual_shared_prefix_len: Any = None,
        query_rope: Any = None,
        key_rope: Any = None,
        key_rope_antiquant_scale: Any = None,
        num_heads: int = 1,
        scale: float = 1.0,
        pre_tokens: int = 2147483647,
        next_tokens: int = 2147483647,
        input_layout: str = "BSH",
        num_key_value_heads: int = 0,
        sparse_mode: int = 0,
        inner_precise: int = 0,
        block_size: int = 0,
        antiquant_mode: int = 0,
        softmax_lse_flag: bool = False,
        key_antiquant_mode: int = 0,
        value_antiquant_mode: int = 0,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        key_list = [key]
        value_list = [value]
        return ge.FusedInferAttentionScore(
            query, key_list, value_list,
            pse_shift=pse_shift,
            atten_mask=atten_mask,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            dequant_scale1=dequant_scale1,
            quant_scale1=quant_scale1,
            dequant_scale2=dequant_scale2,
            quant_scale2=quant_scale2,
            quant_offset2=quant_offset2,
            antiquant_scale=antiquant_scale,
            antiquant_offset=antiquant_offset,
            block_table=block_table,
            query_padding_size=query_padding_size,
            kv_padding_size=kv_padding_size,
            key_antiquant_scale=key_antiquant_scale,
            key_antiquant_offset=key_antiquant_offset,
            value_antiquant_scale=value_antiquant_scale,
            value_antiquant_offset=value_antiquant_offset,
            key_shared_prefix=key_shared_prefix,
            value_shared_prefix=value_shared_prefix,
            actual_shared_prefix_len=actual_shared_prefix_len,
            query_rope=query_rope,
            key_rope=key_rope,
            key_rope_antiquant_scale=key_rope_antiquant_scale,
            num_heads=num_heads,
            scale=scale,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            input_layout=input_layout,
            num_key_value_heads=num_key_value_heads,
            sparse_mode=sparse_mode,
            inner_precise=inner_precise,
            block_size=block_size,
            antiquant_mode=antiquant_mode,
            softmax_lse_flag=softmax_lse_flag,
            key_antiquant_mode=key_antiquant_mode,
            value_antiquant_mode=value_antiquant_mode,
            dequant_scale_query=None,
            learnable_sink=None,
            q_start_idx=None,
            kv_start_idx=None,
            query_quant_mode=0,
            pse_type=0,
            out_dtype=0,
        )

BATCH_SIZE = 128
SEQ_LEN = 2
HIDDEN_DIM = 128
NUM_HEADS = 4
VOCAB_SIZE = 128


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but CUDA is not available")
    return torch.device(device_name)


class ExportSampleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, dtype=torch.bfloat16)
        self.logit_proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, dtype=torch.bfloat16)
        self.scale_value = 1.0 / math.sqrt(HIDDEN_DIM // NUM_HEADS)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sin_cache: torch.Tensor,
        cos_cache: torch.Tensor,
        attn_mask: torch.Tensor,
        prefix_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.ops.cann_lm.rope_in_place(
            hidden_states,
            sin_cache,
            cos_cache,
        )
        rope_output = hidden_states
        query = self.q_proj(rope_output)
        key = self.k_proj(rope_output)
        value = self.v_proj(rope_output)
        attn_output, _ = torch.ops.cann_lm.fused_infer_attention_score(
            query,
            key,
            value,
            atten_mask=attn_mask,
            num_heads=NUM_HEADS,
            scale=self.scale_value,
            input_layout="BSH",
        )
        logits = self.logit_proj(attn_output[:, -1, :])
        top_tokens = torch.empty_like(logits, dtype=torch.int32)
        top_probs = logits + prefix_scores.unsqueeze(-1)
        return torch.ops.cann_lm.beam_search(
            logits,
            top_tokens,
            top_probs,
        )


def build_sample_inputs(device: torch.device | str) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    device = torch.device(device)
    hidden_states = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, device=device, dtype=torch.bfloat16)
    sin_cache = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=torch.bfloat16)
    cos_cache = torch.randn(BATCH_SIZE, HIDDEN_DIM, device=device, dtype=torch.bfloat16)
    causal_mask = torch.triu(
        torch.full((SEQ_LEN, SEQ_LEN), 1, dtype=torch.uint8, device=device),
        diagonal=1,
    )
    attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(BATCH_SIZE, 1, SEQ_LEN, SEQ_LEN)
    prefix_scores = torch.zeros(BATCH_SIZE, device=device)
    return hidden_states, sin_cache, cos_cache, attn_mask, prefix_scores


def export_sample_program(
    output_path: Path,
    device_name: str = "auto",
    print_graph: bool = False,
) -> tuple[Path, torch.export.ExportedProgram]:
    device = select_device(device_name)
    model = ExportSampleModel().eval().to(device)
    sample_inputs = build_sample_inputs(device)
    exported_program = torch.export.export(model, sample_inputs, strict=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported_program, output_path)

    graph_path = output_path.with_suffix(".graph.py")
    print(exported_program)
    graph_path.write_text(str(exported_program.graph_module.graph), encoding="utf-8")

    return output_path, exported_program


def compile_ep_to_epair(
    ep_path: str | Path,
    epair_path: str | Path,
    rewrite_only: bool = False,
) -> Path:
    import torch_delegate

    epair_path = Path(epair_path).resolve()
    output_dir = epair_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_program = torch.export.load(Path(ep_path).resolve())

    rewritten_path = output_dir / f"{epair_path.stem}.rewritten.pt2"
    torch.export.save(exported_program, rewritten_path)
    rewritten_path.with_suffix(".graph.py").write_text(
        exported_program.graph_module.code,
        encoding="utf-8",
    )

    if rewrite_only:
        return rewritten_path

    register_ge_converters()
    register_customop_converters()
    epair = torch_delegate.export_to_epair(exported_program)
    torch_delegate.epair_save(epair, epair_path)
    return epair_path


def _add_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        default="generate/sample.pt2",
        help="Path to the exported .pt2 file.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used to materialize example inputs before export.",
    )
    parser.add_argument(
        "--print-graph",
        action="store_true",
        help="Print the exported FX graph to stdout.",
    )


def _add_compile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ep-path",
        default="generate/sample.pt2",
        help="Path to the input .pt2 file.",
    )
    parser.add_argument(
        "--epair-path",
        default="generate/sample.epair",
        help="Path to the output .epair file.",
    )
    parser.add_argument(
        "--rewrite-only",
        action="store_true",
        help="Only rewrite the graph, skip epair compilation.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CANN-LM model export and epair compilation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export ExportedProgram to .pt2")
    _add_export_args(export_parser)

    compile_parser = subparsers.add_parser("compile", help="Compile .pt2 to .epair")
    _add_compile_args(compile_parser)

    args = parser.parse_args()

    if args.command == "export":
        output_path = Path(args.output).resolve()
        exported_path, _ = export_sample_program(
            output_path=output_path,
            device_name=args.device,
            print_graph=args.print_graph,
        )
        print(f"Exported program saved to: {exported_path}")
        print(f"Graph text saved to: {exported_path.with_suffix('.graph.py')}")

    elif args.command == "compile":
        output_path = compile_ep_to_epair(
            ep_path=args.ep_path,
            epair_path=args.epair_path,
            rewrite_only=args.rewrite_only,
        )
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
