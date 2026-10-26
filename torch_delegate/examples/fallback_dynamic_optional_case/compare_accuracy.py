#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Fallback Dynamic/Optional 测试 - 精度对比脚本

对比 PyTorch 参考输出和 delegate 输出的精度
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custom_ops  # noqa: F401


def load_tensor(output_dir, prefix, name):
    bin_path = output_dir / f"input_{prefix}_{name}.bin"
    shape_path = output_dir / f"input_{prefix}_{name}_shape.txt"
    with open(shape_path) as f:
        shape = tuple(int(d) for d in f.read().strip().split(","))
    return torch.from_numpy(np.fromfile(bin_path, dtype=np.float32).reshape(shape))


def load_delegate_output(output_dir, prefix, idx=0):
    out_path = output_dir / f"delegate_output_{prefix}_{idx}.bin"
    out_shape_path = output_dir / f"delegate_output_{prefix}_{idx}_shape.txt"
    if not out_path.exists():
        return None
    with open(out_shape_path) as f:
        shape = tuple(int(d) for d in f.read().strip().split(","))
    return torch.from_numpy(np.fromfile(out_path, dtype=np.float32).reshape(shape))


def compare_tensors(ref, test, rtol=1e-3, atol=1e-5):
    if ref.shape != test.shape:
        print(
            f"    [WARN] 形状不匹配: ref {list(ref.shape)} vs test {list(test.shape)}"
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
        f"    FAIL: {mismatch_count}/{ref.numel()} 个元素超过容差 (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})"
    )
    return False


def main():
    print("=" * 60)
    print("Fallback Dynamic/Optional 测试 - 精度对比")
    print("=" * 60)

    output_dir = Path(__file__).parent / "output"
    all_pass = True

    print("\n测试 model_addn (Tensor[] dynamic input):")
    addn_x = load_tensor(output_dir, "addn", "x")
    addn_y = load_tensor(output_dir, "addn", "y")
    with torch.no_grad():
        addn_ref = torch.ops.my_ops.my_addn([addn_x, addn_y])
    addn_del = load_delegate_output(output_dir, "addn")
    if addn_del is None:
        print("    [SKIP] delegate 输出未找到")
        all_pass = False
    else:
        print(
            f"  ref shape: {list(addn_ref.shape)}, delegate shape: {list(addn_del.shape)}"
        )
        if not compare_tensors(addn_ref, addn_del):
            all_pass = False

    print("\n测试 model_addmm_no_bias (Tensor? = None, unoccupied):")
    mm_input = load_tensor(output_dir, "addmm_no_bias", "input")
    mm_mat1 = load_tensor(output_dir, "addmm_no_bias", "mat1")
    mm_mat2 = load_tensor(output_dir, "addmm_no_bias", "mat2")
    with torch.no_grad():
        mm_nobias_ref = torch.ops.my_ops.my_addmm(
            mm_input, mm_mat1, mm_mat2, None, beta=2.0
        )
    mm_nobias_del = load_delegate_output(output_dir, "addmm_no_bias")
    if mm_nobias_del is None:
        print("    [SKIP] delegate 输出未找到")
        all_pass = False
    else:
        print(
            f"  ref shape: {list(mm_nobias_ref.shape)}, delegate shape: {list(mm_nobias_del.shape)}"
        )
        if not compare_tensors(mm_nobias_ref, mm_nobias_del):
            all_pass = False

    print("\n测试 model_addmm_with_bias (Tensor? = value, occupied):")
    mmwb_input = load_tensor(output_dir, "addmm_with_bias", "input")
    mmwb_mat1 = load_tensor(output_dir, "addmm_with_bias", "mat1")
    mmwb_mat2 = load_tensor(output_dir, "addmm_with_bias", "mat2")
    ep_path = output_dir / "model_addmm_with_bias.pt2"
    if not ep_path.exists():
        print("    [SKIP] model_addmm_with_bias.pt2 不存在")
        all_pass = False
    else:
        ep = torch.export.load(ep_path)
        with torch.no_grad():
            mmwb_ref = ep.module()(mmwb_input, mmwb_mat1, mmwb_mat2)
        mmwb_del = load_delegate_output(output_dir, "addmm_with_bias")
        if mmwb_del is None:
            print("    [SKIP] delegate 输出未找到")
            all_pass = False
        else:
            print(
                f"  ref shape: {list(mmwb_ref.shape)}, delegate shape: {list(mmwb_del.shape)}"
            )
            if not compare_tensors(mmwb_ref, mmwb_del):
                all_pass = False

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    if all_pass:
        print("✓ 所有测试通过!")
        sys.exit(0)
    else:
        print("✗ 部分测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
