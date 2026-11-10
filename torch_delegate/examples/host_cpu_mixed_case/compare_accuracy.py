#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
HostCpu 混合调度示例 - 精度对比

delegate 输出与 eager CPU 基线逐元素对比。
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custom_ops  # noqa: F401,E402


def load_tensor(path: Path, shape) -> torch.Tensor:
    data = np.fromfile(path, dtype=np.float32)
    return torch.from_numpy(data).reshape(shape)


def compare(ref: torch.Tensor, test: torch.Tensor, rtol=1e-3, atol=1e-5) -> bool:
    if ref.shape != test.shape:
        print(
            f"    [FAIL] Shape mismatch: ref {list(ref.shape)} vs test {list(test.shape)}"
        )
        return False
    abs_diff = (ref - test).abs()
    rel_diff = abs_diff / (ref.abs() + 1e-10)
    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    if max_abs <= atol or max_rel <= rtol:
        print(f"    PASS (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})")
        return True
    print(f"    FAIL: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
    return False


def main() -> None:
    output_dir = Path(__file__).parent / "output"
    # (输入 shape, 输出 shape)：recommender 的输入是 [4] 索引（float32 传入），输出仍为 [4, 8]
    shapes = {
        "model_mixed": ((4, 8), (4, 8)),
        "model_pure_cpu": ((4, 8), (4, 8)),
        "model_non_adjacent": ((4, 8), (4, 8)),
        "model_recommender": ((4,), (4, 8)),
    }
    all_pass = True
    for name, (in_shape, out_shape) in shapes.items():
        print(f"\nTesting {name}:")
        ref = load_tensor(
            output_dir / f"ref_{name.removeprefix('model_')}.bin", out_shape
        )
        out_path = output_dir / f"delegate_output_{name.removeprefix('model_')}.bin"
        if not out_path.exists():
            print("    [SKIP] delegate output not found (runner not executed)")
            all_pass = False
            continue
        test = load_tensor(out_path, out_shape)
        if not compare(ref, test):
            all_pass = False

    print(
        "\n"
        + (
            "[PASS] All models match"
            if all_pass
            else "[FAIL] Accuracy mismatch detected"
        )
    )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
