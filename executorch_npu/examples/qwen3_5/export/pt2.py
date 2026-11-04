# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Export Qwen3.5 prefill and decode PT2 programs."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from examples.auto import AutoModelForCausalLM
from examples.common.export import save_pt2

from . import adapter


def export_split(
    model,
    *,
    prefill_path: Path,
    decode_path: Path,
    trace_len: int,
    max_seq_len: int,
    device: torch.device,
    dynamic_prefill: bool,
    timings: dict | None = None,
) -> None:
    dtype = torch.bfloat16
    timings = {} if timings is None else timings
    timings["prefill"], timings["decode"] = {}, {}
    prefill_wrapper = adapter.ExportWrapper(
        model,
        adapter.create_cache(
            model.config, max_seq_len=max_seq_len, device=device, dtype=dtype
        ),
    ).eval()
    input_ids = torch.zeros((1, trace_len), dtype=torch.long, device=device)
    positions = torch.arange(trace_len, dtype=torch.long, device=device)
    dynamic_shapes = None
    if dynamic_prefill:
        length = torch.export.Dim("sequence_length", min=4, max=max_seq_len - 1)
        dynamic_shapes = {
            "input_ids": {1: length},
            "cache_position": {0: length},
            "full_mask": {2: length},
        }
    full_mask = adapter.build_causal_mask(
        trace_len, max_seq_len, device=device, dtype=dtype
    )
    save_pt2(
        prefill_wrapper,
        {"input_ids": input_ids, "cache_position": positions, "full_mask": full_mask},
        prefill_path,
        dynamic_shapes,
        timings=timings["prefill"],
    )

    decode_wrapper = adapter.ExportWrapper(
        model,
        adapter.create_cache(
            model.config,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            decode=True,
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
        decode_path,
        timings=timings["decode"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("split",), default="split")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            os.environ.get("MODEL_PATH", "/mnt/workspace/models/Qwen3.5-0.8B")
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--max-seq-len", type=int, default=int(os.environ.get("MAX_SEQ_LEN", "2048"))
    )
    parser.add_argument(
        "--trace-seq-len", type=int, default=int(os.environ.get("PREFILL_SEQ_LEN", "8"))
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument(
        "--dynamic-prefill",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DYNAMIC_PREFILL", "1") != "0",
    )
    args = parser.parse_args()
    if args.max_seq_len < 5 or not 4 <= args.trace_seq_len < args.max_seq_len:
        raise ValueError("trace sequence length must be within [4, MAX_SEQ_LEN - 1]")
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    started = time.perf_counter()
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
    if args.output_dir is None:
        prefill_path = Path(
            os.environ.get("PREFILL_PT2_PATH", "/tmp/qwen3_5_prefill.pt2")
        )
        decode_path = Path(os.environ.get("DECODE_PT2_PATH", "/tmp/qwen3_5_decode.pt2"))
    else:
        prefill_path = args.output_dir / "prefill.pt2"
        decode_path = args.output_dir / "decode.pt2"
    timings = {"model_load_s": time.perf_counter() - started}
    export_split(
        model,
        prefill_path=prefill_path,
        decode_path=decode_path,
        trace_len=args.trace_seq_len,
        max_seq_len=args.max_seq_len,
        device=device,
        dynamic_prefill=args.dynamic_prefill,
        timings=timings,
    )

    timings["total_load_capture_save_s"] = time.perf_counter() - started
    (prefill_path.parent / "pt2_timings.json").write_text(
        json.dumps(timings, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
