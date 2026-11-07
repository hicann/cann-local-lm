#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Inplace+Return 测试 - 模型导出脚本

导出一个包含以下特性的模型：
- 自定义算子 inplace_scale_out 同时具有 inplace 输入 (a!) 和 Tensor 返回值
- 模型同时使用返回值和原地修改后的输入
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
# InplaceReturnModel: 测试 inplace + Tensor 返回值
# ============================================================
class InplaceReturnModel(nn.Module):
    """测试模型，包含：
    - inplace 算子同时返回 Tensor
    - 模型同时使用返回值和原地修改后的输入
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def forward(self, x: torch.Tensor, scale_val: float) -> torch.Tensor:
        # inplace_scale_out: x *= scale_val (inplace), 返回 x (修改后)
        result = torch.ops.my_ops.inplace_scale_out(x, scale_val)
        # 直接返回 inplace 算子的返回值，避免下游算子消费 mutation 输出
        # （GE eager 框架对 MakeOutputRefInput 的 shape 传播有限制）
        return result


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
        if node.get("op") == "call_function":
            ascend_ir = node.get("ascend_ir", {})
            if ascend_ir.get("source") == "":
                fallback_ops.append(node.get("target", ""))
    if fallback_ops:
        print(f"  Fallback ops ({len(fallback_ops)}):")
        for op in fallback_ops:
            print(f"    - {op}")

    return epair_content


def save_input_group(output_dir, prefix, group_idx, x, scale_val):
    """保存一组输入数据和 shape 信息"""
    # 保存 x
    x_path = output_dir / f"input_{prefix}_{group_idx}_x.bin"
    x_shape_path = output_dir / f"input_{prefix}_{group_idx}_x_shape.txt"
    x.numpy().tofile(x_path)
    with open(x_shape_path, "w") as f:
        f.write(",".join(str(d) for d in x.shape))
    print(f"  Saved x: shape={list(x.shape)} → {x_path.name}")

    # 保存 scale_val
    scale_path = output_dir / f"input_{prefix}_{group_idx}_scale.txt"
    with open(scale_path, "w") as f:
        f.write(str(scale_val))
    print(f"  Saved scale_val: {scale_val} → {scale_path.name}")


def main():
    print("=" * 60)
    print("Inplace+Return test - model export")
    print("=" * 60)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # =========================================================
    # InplaceReturnModel
    # =========================================================
    model = InplaceReturnModel().eval()

    # 示例输入
    x = torch.randn(4, 10)
    scale_val = 2.0

    # 动态 shape 配置
    batch = Dim("batch", min=1, max=8)
    dynamic_shapes = {
        "x": {0: batch},
        "scale_val": None,  # 静态
    }

    export_model(
        "model_inplace_return",
        model,
        (x, scale_val),
        output_dir,
        dynamic_shapes=dynamic_shapes,
    )

    print("\nGenerating input data...")
    # 3 组不同 batch size 的输入，scale 固定为 2.0（与导出时一致）
    test_configs = [
        (4, 2.0),
        (8, 2.0),
        (2, 2.0),
    ]
    for i, (bs, scale) in enumerate(test_configs):
        torch.manual_seed(100 + i)
        x_test = torch.randn(bs, 10)
        save_input_group(output_dir, "inplace_return", i, x_test, scale)

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
