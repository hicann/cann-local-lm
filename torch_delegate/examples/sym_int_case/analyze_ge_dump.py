#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
分析 GE dump 数据，验证 converter 节点（Sigmoid, Tanh, Add）执行正确
"""

import sys
from pathlib import Path
import numpy as np
import torch

# 添加 custom_ops 到路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 加载输入数据
output_dir = Path("output")
ge_dump_dir = Path("ge_dump/20260611145230/0/epair_graph/1")


def load_input(group_idx):
    """加载输入数据"""
    x_bin = output_dir / f"input_simple_{group_idx}_x.bin"
    x_shape = output_dir / f"input_simple_{group_idx}_x_shape.txt"
    st_bin = output_dir / f"input_simple_{group_idx}_static_tensor.bin"
    st_shape = output_dir / f"input_simple_{group_idx}_static_tensor_shape.txt"

    with open(x_shape) as f:
        x_shape = tuple(int(d) for d in f.read().strip().split(","))
    with open(st_shape) as f:
        st_shape = tuple(int(d) for d in f.read().strip().split(","))

    x = torch.from_numpy(np.fromfile(x_bin, dtype=np.float32).reshape(x_shape))
    static_tensor = torch.from_numpy(
        np.fromfile(st_bin, dtype=np.float32).reshape(st_shape)
    )

    return x, static_tensor


def load_ge_dump(group_idx, op_name):
    """加载 GE dump 数据"""
    dump_path = ge_dump_dir / str(group_idx)
    npy_files = list(dump_path.glob(f"{op_name}.*.npy"))

    data = {}
    for f in npy_files:
        parts = f.stem.split(".")
        if "input" in parts:
            idx = int(parts[-1])
            data[f"input_{idx}"] = np.load(f)
        elif "output" in parts:
            idx = int(parts[-1])
            data[f"output_{idx}"] = np.load(f)

    return data


def compare(name, ref, actual, rtol=1e-3, atol=1e-5):
    """比较两个张量"""
    ref_np = ref.numpy() if isinstance(ref, torch.Tensor) else ref
    actual_np = actual.numpy() if isinstance(actual, torch.Tensor) else actual

    if ref_np.shape != actual_np.shape:
        print(f"  {name}: FAIL - shape mismatch {ref_np.shape} vs {actual_np.shape}")
        return False

    abs_diff = np.abs(ref_np - actual_np)
    rel_diff = abs_diff / (np.abs(ref_np) + 1e-10)

    max_abs = abs_diff.max()
    max_rel = rel_diff.max()

    if max_abs <= atol or max_rel <= rtol:
        print(f"  {name}: PASS (max_abs={max_abs:.2e}, max_rel={max_rel:.2e})")
        return True

    mismatch = (abs_diff > atol) & (rel_diff > rtol)
    mismatch_count = mismatch.sum()
    print(f"  {name}: FAIL - {mismatch_count}/{ref_np.size} elements exceed tolerance")
    print(f"    max_abs={max_abs:.2e}, max_rel={max_rel:.2e}")
    print(
        f"    ref: min={ref_np.min():.4f}, max={ref_np.max():.4f}, mean={ref_np.mean():.4f}"
    )
    print(
        f"    actual: min={actual_np.min():.4f}, max={actual_np.max():.4f}, mean={actual_np.mean():.4f}"
    )
    return False


print("=" * 60)
print("Analyzing GE dump data")
print("=" * 60)

for group_idx in range(3):
    print(f"\nInput group {group_idx}:")

    # 加载输入
    x, static_tensor = load_input(group_idx)
    print(f"  x: shape={list(x.shape)}, min={x.min():.4f}, max={x.max():.4f}")
    print(
        f"  static_tensor: shape={list(static_tensor.shape)}, min={static_tensor.min():.4f}, max={static_tensor.max():.4f}"
    )

    # PyTorch 参考计算
    with torch.no_grad():
        sigmoid_ref = torch.sigmoid(x)
        tanh_ref = torch.tanh(sigmoid_ref)
        add_ref = tanh_ref + static_tensor

    # 加载 GE dump
    sigmoid_dump = load_ge_dump(group_idx, "Sigmoid")
    tanh_dump = load_ge_dump(group_idx, "Tanh")
    add_dump = load_ge_dump(group_idx, "Add")

    # 比较 Sigmoid
    if "input_0" in sigmoid_dump:
        compare("Sigmoid input", x, sigmoid_dump["input_0"])
    if "output_0" in sigmoid_dump:
        compare("Sigmoid output", sigmoid_ref, sigmoid_dump["output_0"])

    # 比较 Tanh
    if "input_0" in tanh_dump:
        compare("Tanh input", sigmoid_ref, tanh_dump["input_0"])
    if "output_0" in tanh_dump:
        compare("Tanh output", tanh_ref, tanh_dump["output_0"])

    # 比较 Add
    if "input_0" in add_dump:
        compare("Add input[0]", tanh_ref, add_dump["input_0"])
    if "input_1" in add_dump:
        compare("Add input[1]", static_tensor, add_dump["input_1"])
    if "output_0" in add_dump:
        compare("Add output", add_ref, add_dump["output_0"])

print("\n" + "=" * 60)
