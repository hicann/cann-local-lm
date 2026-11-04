# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Export Gemma4 prefill/decode or unified PT2 programs."""

import argparse
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from examples.auto import AutoModelForCausalLM
from examples.common.export import save_pt2

from . import adapter


def export_split(
    model,
    *,
    output_dir: Path,
    trace_len: int,
    max_seq_len: int,
    device: torch.device,
    dynamic_prefill: bool,
) -> None:
    dtype = torch.bfloat16
    wrapper = adapter.ExportWrapper(
        model,
        adapter.create_cache(
            model.config, max_seq_len=max_seq_len, device=device, dtype=dtype
        ),
    ).eval()
    input_ids = torch.zeros((1, trace_len), dtype=torch.long, device=device)
    positions = torch.arange(trace_len, dtype=torch.long, device=device)
    dynamic_shapes = None
    if dynamic_prefill:
        length = torch.export.Dim("sequence_length", min=2, max=max_seq_len)
        dynamic_shapes = {
            "input_ids": {1: length},
            "cache_position": {0: length},
            "sliding_mask": {2: length},
            "full_mask": {2: length},
        }
    text_config = model.config.get_text_config(decoder=True)
    window = (
        getattr(text_config, "sliding_window", None)
        if "sliding_attention" in wrapper.layer_types
        else None
    )
    sliding = adapter.build_mask(
        trace_len, max_seq_len, window, device=device, dtype=dtype
    )
    full = adapter.build_mask(trace_len, max_seq_len, None, device=device, dtype=dtype)
    save_pt2(
        wrapper,
        {
            "input_ids": input_ids,
            "cache_position": positions,
            "sliding_mask": sliding,
            "full_mask": full,
        },
        output_dir / "prefill.pt2",
        dynamic_shapes,
    )

    decode_wrapper = adapter.ExportWrapper(
        model,
        adapter.create_cache(
            model.config, max_seq_len=max_seq_len, device=device, dtype=dtype
        ),
    ).eval()
    save_pt2(
        decode_wrapper,
        {
            "input_ids": torch.zeros((1, 1), dtype=torch.long, device=device),
            "cache_position": torch.tensor(
                [trace_len], dtype=torch.long, device=device
            ),
        },
        output_dir / "decode.pt2",
    )


def export_unified(
    model,
    *,
    output: Path,
    trace_len: int,
    max_seq_len: int,
    device: torch.device,
    dynamic_shapes_enabled: bool,
) -> None:
    wrapper = adapter.UnifiedExportWrapper(
        model,
        adapter.create_cache(
            model.config, max_seq_len=max_seq_len, device=device, dtype=torch.bfloat16
        ),
    ).eval()
    shapes = None
    if dynamic_shapes_enabled:
        length = torch.export.Dim("sequence_length", min=2, max=max_seq_len)
        shapes = {"input_ids": {1: length}, "cache_position": {0: length}}
    save_pt2(
        wrapper,
        {
            "input_ids": torch.zeros((1, trace_len), dtype=torch.long, device=device),
            "cache_position": torch.arange(trace_len, dtype=torch.long, device=device),
        },
        output,
        shapes,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("split", "unified"), default="split")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            os.environ.get("MODEL_PATH", "/mnt/workspace/models/gemma-4-E2B-it")
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--trace-seq-len", type=int)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument(
        "--dynamic-prefill", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args()
    max_seq_len = args.max_seq_len
    if max_seq_len is None:
        legacy_default = "2048" if args.mode == "unified" else "8"
        max_seq_len = int(os.environ.get("MAX_SEQ_LEN", legacy_default))
    default_trace = "TRACE_SEQ_LEN" if args.mode == "unified" else "PREFILL_SEQ_LEN"
    trace_len = args.trace_seq_len
    if trace_len is None:
        trace_len = int(
            os.environ.get(default_trace, "8" if args.mode == "unified" else "4")
        )
    minimum = 2
    if max_seq_len < minimum or not minimum <= trace_len <= max_seq_len:
        raise ValueError("trace sequence length must be within [2, MAX_SEQ_LEN]")
    enabled = args.dynamic_prefill
    if enabled is None:
        key = "DYNAMIC_SHAPES" if args.mode == "unified" else "DYNAMIC_PREFILL"
        enabled = os.environ.get(key, "1") != "0"
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            strict=True,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    model.requires_grad_(False)
    if args.mode == "split":
        output_dir = args.output_dir or Path(
            os.environ.get("OUT_ROOT", "/mnt/workspace/scratch_gemma4_pd")
        )
        export_split(
            model,
            output_dir=output_dir,
            trace_len=trace_len,
            max_seq_len=max_seq_len,
            device=device,
            dynamic_prefill=enabled,
        )
    else:
        output = (
            args.output_dir / "model.pt2"
            if args.output_dir is not None
            else Path(os.environ.get("PT2_PATH", "/tmp/gemma4_model.pt2"))
        )
        export_unified(
            model,
            output=output,
            trace_len=trace_len,
            max_seq_len=max_seq_len,
            device=device,
            dynamic_shapes_enabled=enabled,
        )


if __name__ == "__main__":
    main()
