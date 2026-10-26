#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Inplace+Return 测试 - 精度对比脚本

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
# InplaceReturnModel: 与 export_model.py 保持一致
# ============================================================
class InplaceReturnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def forward(self, x: torch.Tensor, scale_val: float) -> torch.Tensor:
        result = torch.ops.my_ops.inplace_scale_out(x, scale_val)
        return result


def load_input_group(output_dir, prefix, group_idx):
    """加载一组输入数据"""
    x_path = output_dir / f"input_{prefix}_{group_idx}_x.bin"
    x_shape_path = output_dir / f"input_{prefix}_{group_idx}_x_shape.txt"

    with open(x_shape_path) as f:
        x_shape = tuple(int(d) for d in f.read().strip().split(","))
    x = torch.from_numpy(np.fromfile(x_path, dtype=np.float32).reshape(x_shape))

    scale_path = output_dir / f"input_{prefix}_{group_idx}_scale.txt"
    with open(scale_path) as f:
        scale_val = float(f.read().strip())

    return x, scale_val


def find_delegate_outputs(output_dir, prefix, group_idx):
    """查找所有 delegate 输出文件"""
    outputs = {}
    for i in range(100):
        out_path = output_dir / f"delegate_output_{prefix}_{group_idx}_{i}.bin"
        if not out_path.exists():
            break
        out_shape_path = (
            output_dir / f"delegate_output_{prefix}_{group_idx}_{i}_shape.txt"
        )
        with open(out_shape_path) as f:
            content = f.read().strip()
            out_shape = tuple(int(d) for d in content.split(",")) if content else ()
        outputs[i] = torch.from_numpy(
            np.fromfile(out_path, dtype=np.float32).reshape(out_shape)
        )
    return outputs


def compare_tensors(ref, test, rtol=1e-3, atol=1e-5):
    """比较两个张量"""
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
    print("Inplace+Return 测试 - 精度对比")
    print("=" * 60)

    output_dir = Path(__file__).parent / "output"

    print("\n" + "=" * 60)
    print("测试 model_inplace_return")
    print("=" * 60)

    model = InplaceReturnModel().eval()

    test_configs = [
        (4, 2.0),
        (8, 2.0),
        (2, 2.0),
    ]
    all_pass = True

    for i, (bs, scale) in enumerate(test_configs):
        print(f"\n输入组 {i} (batch_size={bs}, scale={scale}):")

        x, scale_val = load_input_group(output_dir, "inplace_return", i)

        # PyTorch 参考输出（需要 clone，因为 inplace 会修改输入）
        x_ref = x.clone()
        with torch.no_grad():
            ref_output = model(x_ref, scale_val)

        # 查找所有 delegate 输出
        delegate_outputs = find_delegate_outputs(output_dir, "inplace_return", i)
        if not delegate_outputs:
            print("    [SKIP] delegate 输出未找到")
            all_pass = False
            continue

        print(f"    delegate 输出数量: {len(delegate_outputs)}")
        for idx, out in sorted(delegate_outputs.items()):
            print(
                f"      output[{idx}]: shape={list(out.shape)}, "
                f"min={out.min().item():.4f}, max={out.max().item():.4f}"
            )

        # 比较所有输出，找到与 PyTorch 参考匹配的
        matched = False
        for idx in sorted(delegate_outputs.keys()):
            if delegate_outputs[idx].shape == ref_output.shape:
                print(f"    比较 output[{idx}] 与 PyTorch 参考:")
                if compare_tensors(ref_output, delegate_outputs[idx]):
                    matched = True
                    break
        if not matched:
            print(
                f"    [FAIL] 没有输出与 PyTorch 参考匹配 (ref shape={list(ref_output.shape)})"
            )
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
