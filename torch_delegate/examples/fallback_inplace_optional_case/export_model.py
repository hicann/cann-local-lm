#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Fallback Inplace/Optional 测试 - 模型导出脚本

导出一个包含以下特性的模型：
- 多 inplace 参数算子 (dual_inplace_)
- 多可选参数算子 (multi_optional_op)
  - 场景 1: 只传 gamma
  - 场景 2: 传 alpha + gamma
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.export import Dim

# 注册自定义算子
sys.path.insert(0, str(Path(__file__).resolve().parent))
import custom_ops  # noqa: F401

import torch_delegate as td

import logging

try:
    from torchair.core.utils import logger

    logger.setLevel(logging.WARN)
except ImportError:
    pass


# ============================================================
# InplaceOptionalModel: 包含所有测试场景
# ============================================================
class InplaceOptionalModel(nn.Module):
    """测试模型，包含：
    - 多 inplace 参数算子 (dual_inplace_)
    - 多可选参数算子 (multi_optional_op)
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, alpha_val: float, gamma_val: float
    ) -> torch.Tensor:
        # 场景 1: 多 inplace — x 和 y 都被原地修改
        torch.ops.my_ops.dual_inplace_(x, y, 0.5, 2.0)

        # 场景 2: 多可选参数 — 只传 gamma
        result1 = torch.ops.my_ops.multi_optional_op(x, gamma=gamma_val)

        # 场景 3: 多可选参数 — 传 alpha + gamma
        result2 = torch.ops.my_ops.multi_optional_op(
            result1, alpha=alpha_val, gamma=gamma_val
        )

        return result2 + y


# ============================================================
# 导出辅助函数
# ============================================================
def export_model(model_name, model, example_inputs, output_dir, dynamic_shapes=None):
    """导出单个模型到 epair"""
    print(f"\n{'=' * 60}")
    print(f"Exporting {model_name}")
    print(f"{'=' * 60}")

    print("[1/3] Exporting ExportedProgram...")
    if dynamic_shapes is not None:
        ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
        print(f"  Dynamic shape: {dynamic_shapes}")
    else:
        ep = torch.export.export(model, example_inputs)

    ep.graph_module.print_readable()
    ep_path = output_dir / f"{model_name}.pt2"
    torch.export.save(ep, ep_path)
    print(f"  Saved EP: {ep_path}")

    print("[2/3] Exporting epair...")
    epair_content = td.export_to_epair(ep)
    epair_path = output_dir / f"{model_name}.epair"
    td.epair_save(epair_content, epair_path)
    print(f"  Saved epair: {epair_path}")

    print("[3/3] Statistics:")
    total = epair_content.manifest.total_nodes
    converted = epair_content.manifest.converted_nodes
    fallback = total - converted
    print(f"  Total nodes: {total}")
    print(f"  Converted nodes: {converted}")
    print(f"  Fallback nodes: {fallback}")

    fallback_ops = []
    for node in epair_content.ep_json["graph_module"]["graph"]["nodes"]:
        ascend_ir = node.get("ascend_ir", {})
        if ascend_ir.get("source") == "":
            fallback_ops.append(node.get("target", ""))
    if fallback_ops:
        print(f"  Fallback ops ({len(fallback_ops)}):")
        for op in fallback_ops:
            print(f"    - {op}")

    return epair_content


def save_input_group(output_dir, prefix, group_idx, x, y, alpha_val, gamma_val):
    """保存一组输入数据和 shape 信息"""
    # 保存 x
    x_path = output_dir / f"input_{prefix}_{group_idx}_x.bin"
    x_shape_path = output_dir / f"input_{prefix}_{group_idx}_x_shape.txt"
    x.numpy().tofile(x_path)
    with open(x_shape_path, "w") as f:
        f.write(",".join(str(d) for d in x.shape))
    print(f"  Saved x: shape={list(x.shape)} → {x_path.name}")

    # 保存 y
    y_path = output_dir / f"input_{prefix}_{group_idx}_y.bin"
    y_shape_path = output_dir / f"input_{prefix}_{group_idx}_y_shape.txt"
    y.numpy().tofile(y_path)
    with open(y_shape_path, "w") as f:
        f.write(",".join(str(d) for d in y.shape))
    print(f"  Saved y: shape={list(y.shape)} → {y_path.name}")

    # 保存 alpha_val
    alpha_path = output_dir / f"input_{prefix}_{group_idx}_alpha.txt"
    with open(alpha_path, "w") as f:
        f.write(str(alpha_val))
    print(f"  Saved alpha: {alpha_val} → {alpha_path.name}")

    # 保存 gamma_val
    gamma_path = output_dir / f"input_{prefix}_{group_idx}_gamma.txt"
    with open(gamma_path, "w") as f:
        f.write(str(gamma_val))
    print(f"  Saved gamma: {gamma_val} → {gamma_path.name}")


def main():
    print("=" * 60)
    print("Fallback Inplace/Optional test - model export")
    print("=" * 60)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # =========================================================
    # InplaceOptionalModel
    # =========================================================
    model = InplaceOptionalModel().eval()

    # 示例输入
    x = torch.randn(4, 10)
    y = torch.randn(4, 10)
    alpha_val = 2.0
    gamma_val = 3.0

    # 动态 shape 配置
    batch = Dim("batch", min=1, max=8)
    dynamic_shapes = {
        "x": {0: batch},
        "y": {0: batch},
        "alpha_val": None,
        "gamma_val": None,
    }

    export_model(
        "model_inplace_optional",
        model,
        (x, y, alpha_val, gamma_val),
        output_dir,
        dynamic_shapes=dynamic_shapes,
    )

    print("\nGenerating input data...")
    # 3 组不同 batch size 的输入（固定 alpha=2.0, gamma=3.0）
    test_configs = [
        (4, 2.0, 3.0),
        (8, 2.0, 3.0),
        (2, 2.0, 3.0),
    ]
    for i, (bs, alpha, gamma) in enumerate(test_configs):
        torch.manual_seed(100 + i)
        x_test = torch.randn(bs, 10)
        y_test = torch.randn(bs, 10)
        save_input_group(
            output_dir, "inplace_optional", i, x_test, y_test, alpha, gamma
        )

    # =========================================================
    # 总结
    # =========================================================
    print("\n" + "=" * 60)
    print("Export completed!")
    print("=" * 60)
    print("\nOutput files:")
    for f in sorted(output_dir.glob("*.epair")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("input_*.bin")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("input_*.txt")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
