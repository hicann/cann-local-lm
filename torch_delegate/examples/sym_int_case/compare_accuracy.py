#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
E2E 测试 - 精度对比脚本

对比 PyTorch 参考输出和 delegate 输出的精度
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 注册自定义算子
sys.path.insert(0, str(Path(__file__).resolve().parent))
import custom_ops  # noqa: F401


# ============================================================
# SimpleMixedModel: 与 export_model.py 保持一致
# ============================================================
class SimpleMixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def forward(
        self, x: torch.Tensor, static_tensor: torch.Tensor, scale: int, multiplier: int
    ) -> torch.Tensor:
        y = torch.sigmoid(x)
        y = torch.tanh(y)
        y = y + static_tensor
        y = torch.ops.my_ops.scale_add(y, y, 0.5, scale)
        sym_multiplier = torch.sym_int(multiplier)
        y = torch.ops.my_ops.scale_add(y, y, 1.0, sym_multiplier)
        return y


# ============================================================
# SimpleStaticModel: 与 export_model.py 保持一致（纯静态 shape）
# ============================================================
class SimpleStaticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def forward(
        self, x: torch.Tensor, static_tensor: torch.Tensor, scale: int, multiplier: int
    ) -> torch.Tensor:
        y = torch.sigmoid(x)
        y = torch.tanh(y)
        y = y + static_tensor
        y = torch.ops.my_ops.scale_add(y, y, 0.5, scale)
        y = torch.ops.my_ops.scale_add(y, y, 1.0, multiplier)
        return y


def load_input_group(output_dir, prefix, group_idx):
    """加载一组输入数据"""
    x_path = output_dir / f"input_{prefix}_{group_idx}_x.bin"
    x_shape_path = output_dir / f"input_{prefix}_{group_idx}_x_shape.txt"

    with open(x_shape_path) as f:
        x_shape = tuple(int(d) for d in f.read().strip().split(","))
    x = torch.from_numpy(np.fromfile(x_path, dtype=np.float32).reshape(x_shape))

    st_path = output_dir / f"input_{prefix}_{group_idx}_static_tensor.bin"
    st_shape_path = output_dir / f"input_{prefix}_{group_idx}_static_tensor_shape.txt"
    with open(st_shape_path) as f:
        st_shape = tuple(int(d) for d in f.read().strip().split(","))
    static_tensor = torch.from_numpy(
        np.fromfile(st_path, dtype=np.float32).reshape(st_shape)
    )

    scale_path = output_dir / f"input_{prefix}_{group_idx}_scale.txt"
    with open(scale_path) as f:
        scale = int(f.read().strip())

    mult_path = output_dir / f"input_{prefix}_{group_idx}_multiplier.txt"
    with open(mult_path) as f:
        multiplier = int(f.read().strip())

    return x, static_tensor, scale, multiplier


def load_delegate_output(output_dir, prefix, group_idx):
    """加载 delegate 输出"""
    out_path = output_dir / f"delegate_output_{prefix}_{group_idx}_0.bin"
    out_shape_path = output_dir / f"delegate_output_{prefix}_{group_idx}_0_shape.txt"

    if not out_path.exists():
        return None

    with open(out_shape_path) as f:
        out_shape = tuple(int(d) for d in f.read().strip().split(","))
    return torch.from_numpy(np.fromfile(out_path, dtype=np.float32).reshape(out_shape))


def compare_tensors(ref, test, rtol=1e-3, atol=1e-5):
    """比较两个张量"""
    if ref.shape != test.shape:
        print(
            f"    [WARN] Shape mismatch: ref {list(ref.shape)} vs test {list(test.shape)}"
        )
        return False

    abs_diff = (ref - test).abs()
    rel_diff = abs_diff / (ref.abs() + 1e-10)

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()

    if max_abs <= atol or max_rel <= rtol:
        print(f"    PASS (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})")
        return True

    mismatch_mask = (abs_diff > atol) & (rel_diff > rtol)
    mismatch_count = mismatch_mask.sum().item()
    print(
        f"    FAIL: {mismatch_count}/{ref.numel()} elements exceed tolerance (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})"
    )
    return False


def main():
    print("=" * 60)
    print("E2E test - accuracy comparison")
    print("=" * 60)

    output_dir = Path(__file__).parent / "output"

    # =========================================================
    # SimpleMixedModel
    # =========================================================
    print("\n" + "=" * 60)
    print("Testing model_simple")
    print("=" * 60)

    model = SimpleMixedModel().eval()

    # 3 组不同 batch size 的输入
    batch_sizes = [4, 4, 1, 7, 2, 8, 3, 6, 5]
    all_pass = True

    for i, bs in enumerate(batch_sizes):
        print(f"\nInput group {i} (batch_size={bs}):")

        # 加载输入
        x, static_tensor, scale, multiplier = load_input_group(output_dir, "simple", i)

        # PyTorch 参考输出
        with torch.no_grad():
            ref_output = model(x, static_tensor, scale, multiplier)

        # 加载 delegate 输出
        delegate_output = load_delegate_output(output_dir, "simple", i)
        if delegate_output is None:
            print("    [SKIP] model_simple: delegate output not found")
            all_pass = False
            continue

        # 比较
        if not compare_tensors(ref_output, delegate_output):
            all_pass = False

    # =========================================================
    # SimpleStaticModel (静态 shape)
    # =========================================================
    print("\n" + "=" * 60)
    print("Testing model_static (static shape)")
    print("=" * 60)

    static_model = SimpleStaticModel().eval()

    for i in range(3):
        print(f"\nStatic input group {i} (batch_size=4):")

        # 加载输入
        x, static_tensor, scale, multiplier = load_input_group(output_dir, "static", i)

        # PyTorch 参考输出
        with torch.no_grad():
            ref_output = static_model(x, static_tensor, scale, multiplier)

        # 加载 delegate 输出
        delegate_output = load_delegate_output(output_dir, "static", i)
        if delegate_output is None:
            print("    [SKIP] model_static: delegate output not found")
            all_pass = False
            continue

        # 比较
        if not compare_tensors(ref_output, delegate_output):
            all_pass = False

    # =========================================================
    # 总结
    # =========================================================
    print("\n" + "=" * 60)
    print("Test summary")
    print("=" * 60)

    if all_pass:
        print("[PASS] All tests passed!")
        sys.exit(0)
    else:
        print("[FAIL] Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
