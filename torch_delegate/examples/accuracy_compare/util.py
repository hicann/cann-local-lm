# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared utilities for the business export subsystem."""

from __future__ import annotations

import argparse

import torch


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU export requested but NPU is not available")
        return torch.device("npu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but CUDA is not available")
    return torch.device(device_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a sample ExportedProgram with three custom ops.",
    )
    parser.add_argument(
        "--output",
        default="generate/sample.pt2",
        help="Path to the exported .pt2 file.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "npu"),
        default="auto",
        help="Device used to materialize example inputs before export.",
    )
    parser.add_argument(
        "--print-graph",
        action="store_true",
        help="Print the exported FX graph to stdout.",
    )
    return parser.parse_args()
