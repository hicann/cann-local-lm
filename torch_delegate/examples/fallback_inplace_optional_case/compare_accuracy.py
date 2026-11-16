#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Fallback Inplace/Optional 测试 - 精度对比脚本

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
# InplaceOptionalModel: 与 export_model.py 保持一致
# ============================================================
class InplaceOptionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, alpha_val: float, gamma_val: float
    ) -> torch.Tensor:
        torch.ops.my_ops.dual_inplace_(x, y, 0.5, 2.0)
        result1 = torch.ops.my_ops.multi_optional_op(x, gamma=gamma_val)
        result2 = torch.ops.my_ops.multi_optional_op(
            result1, alpha=alpha_val, gamma=gamma_val
        )
        return result2 + y


def load_input_group(output_dir, prefix, group_idx):
    """加载一组输入数据"""
    x_path = output_dir / f"input_{prefix}_{group_idx}_x.bin"
    x_shape_path = output_dir / f"input_{prefix}_{group_idx}_x_shape.txt"

    with open(x_shape_path) as f:
        x_shape = tuple(int(d) for d in f.read().strip().split(","))
    x = torch.from_numpy(np.fromfile(x_path, dtype=np.float32).reshape(x_shape))

    y_path = output_dir / f"input_{prefix}_{group_idx}_y.bin"
    y_shape_path = output_dir / f"input_{prefix}_{group_idx}_y_shape.txt"
    with open(y_shape_path) as f:
        y_shape = tuple(int(d) for d in f.read().strip().split(","))
    y = torch.from_numpy(np.fromfile(y_path, dtype=np.float32).reshape(y_shape))

    alpha_path = output_dir / f"input_{prefix}_{group_idx}_alpha.txt"
    with open(alpha_path) as f:
        alpha_val = float(f.read().strip())

    gamma_path = output_dir / f"input_{prefix}_{group_idx}_gamma.txt"
    with open(gamma_path) as f:
        gamma_val = float(f.read().strip())

    return x, y, alpha_val, gamma_val


def load_delegate_output(output_dir, prefix, group_idx):
    """加载 delegate 输出"""
    out_path = output_dir / f"delegate_output_{prefix}_{group_idx}_2.bin"
    out_shape_path = output_dir / f"delegate_output_{prefix}_{group_idx}_2_shape.txt"

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
    print("Fallback Inplace/Optional test - accuracy comparison")
    print("=" * 60)

    output_dir = Path(__file__).parent / "output"

    # =========================================================
    # InplaceOptionalModel
    # =========================================================
    print("\n" + "=" * 60)
    print("Testing model_inplace_optional")
    print("=" * 60)

    model = InplaceOptionalModel().eval()

    # 3 组不同配置的输入（固定 alpha=2.0, gamma=3.0）
    test_configs = [
        (4, 2.0, 3.0),
        (8, 2.0, 3.0),
        (2, 2.0, 3.0),
    ]
    all_pass = True

    for i, (bs, alpha, gamma) in enumerate(test_configs):
        print(f"\nInput group {i} (batch_size={bs}, alpha={alpha}, gamma={gamma}):")

        # 加载输入
        x, y, alpha_val, gamma_val = load_input_group(output_dir, "inplace_optional", i)

        # PyTorch 参考输出（需要 clone，因为 inplace 会修改输入）
        x_ref = x.clone()
        y_ref = y.clone()
        with torch.no_grad():
            ref_output = model(x_ref, y_ref, alpha_val, gamma_val)

        # 加载 delegate 输出
        delegate_output = load_delegate_output(output_dir, "inplace_optional", i)
        if delegate_output is None:
            print("    [SKIP] delegate output not found")
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
