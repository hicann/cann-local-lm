#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
E2E 测试 - 模型导出脚本

导出一个包含以下特性的模型：
- 动态 tensor 输入 (x, 动态 batch)
- 静态 tensor 输入 (static_tensor)
- 静态非 tensor 输入 (scale, int)
- 动态非 tensor 输入 (batch_size, 来自 x.size(0))
- Converter 算子 (sigmoid, tanh, add)
- Custom 算子 (scale_add, fallback)
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
# SimpleMixedModel: 包含所有测试场景
# ============================================================
class SimpleMixedModel(nn.Module):
    """简化版混合模型，包含：
    - 动态 tensor 输入 (x)
    - 静态 tensor 输入 (static_tensor)
    - 静态非 tensor 输入 (scale)
    - 动态非 tensor 输入 (multiplier, 来自 x.size(0))
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def forward(
        self, x: torch.Tensor, static_tensor: torch.Tensor, scale: int, multiplier: int
    ) -> torch.Tensor:
        # Converter ops (有 converter)
        y = torch.sigmoid(x)
        y = torch.tanh(y)

        # 使用静态 tensor
        y = y + static_tensor

        # Custom op with static non-tensor input
        y = torch.ops.my_ops.scale_add(y, y, 0.5, scale)

        # Custom op with dynamic non-tensor input
        sym_multiplier = torch.sym_int(multiplier)
        y = torch.ops.my_ops.scale_add(y, y, 1.0, sym_multiplier)

        return y


# ============================================================
# SimpleStaticModel: 纯静态 shape 场景
# ============================================================
class SimpleStaticModel(nn.Module):
    """静态 shape 模型，所有输入 shape 固定：
    - 静态 tensor 输入 (x, 固定 batch=4)
    - 静态 tensor 输入 (static_tensor)
    - 静态非 tensor 输入 (scale, int)
    - 静态非 tensor 输入 (multiplier, int，非 SymInt)
    - Converter 算子 (sigmoid, tanh, add)
    - Custom 算子 (scale_add, fallback)
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def forward(
        self, x: torch.Tensor, static_tensor: torch.Tensor, scale: int, multiplier: int
    ) -> torch.Tensor:
        # Converter ops (有 converter)
        y = torch.sigmoid(x)
        y = torch.tanh(y)

        # 使用静态 tensor
        y = y + static_tensor

        # Custom op with static non-tensor input
        y = torch.ops.my_ops.scale_add(y, y, 0.5, scale)

        # Custom op with static non-tensor input (普通 int，非 SymInt)
        y = torch.ops.my_ops.scale_add(y, y, 1.0, multiplier)

        return y


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


def save_input_group(
    output_dir, prefix, group_idx, x, static_tensor, scale, multiplier
):
    """保存一组输入数据和 shape 信息"""
    # 保存 x
    x_path = output_dir / f"input_{prefix}_{group_idx}_x.bin"
    x_shape_path = output_dir / f"input_{prefix}_{group_idx}_x_shape.txt"
    x.numpy().tofile(x_path)
    with open(x_shape_path, "w") as f:
        f.write(",".join(str(d) for d in x.shape))
    print(f"  Saved x: shape={list(x.shape)} → {x_path.name}")

    # 保存 static_tensor
    st_path = output_dir / f"input_{prefix}_{group_idx}_static_tensor.bin"
    st_shape_path = output_dir / f"input_{prefix}_{group_idx}_static_tensor_shape.txt"
    static_tensor.numpy().tofile(st_path)
    with open(st_shape_path, "w") as f:
        f.write(",".join(str(d) for d in static_tensor.shape))
    print(f"  Saved static_tensor: shape={list(static_tensor.shape)} → {st_path.name}")

    # 保存 scale
    scale_path = output_dir / f"input_{prefix}_{group_idx}_scale.txt"
    with open(scale_path, "w") as f:
        f.write(str(scale))
    print(f"  Saved scale: {scale} → {scale_path.name}")

    # 保存 multiplier
    mult_path = output_dir / f"input_{prefix}_{group_idx}_multiplier.txt"
    with open(mult_path, "w") as f:
        f.write(str(multiplier))
    print(f"  Saved multiplier: {multiplier} → {mult_path.name}")


def main():
    print("=" * 60)
    print("E2E test - model export")
    print("=" * 60)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # =========================================================
    # SimpleMixedModel
    # =========================================================
    model = SimpleMixedModel().eval()

    # 示例输入
    x = torch.randn(4, 10)  # 使用较大的 batch size 避免特化
    static_tensor = torch.randn(10)
    scale = 2  # 静态非 tensor
    multiplier = 4  # 动态非 tensor

    # 动态 shape 配置
    batch = Dim("batch", min=1, max=8)
    dynamic_shapes = {
        "x": {0: batch},
        "static_tensor": None,  # 静态
        "scale": None,  # 静态
        "multiplier": Dim.AUTO,  # 动态
    }

    export_model(
        "model_simple",
        model,
        (x, static_tensor, scale, multiplier),
        output_dir,
        dynamic_shapes=dynamic_shapes,
    )

    print("\nGenerating input data...")
    # 3 组不同 batch size 的输入
    batch_sizes = [4, 4, 1, 7, 2, 8, 3, 6, 5]
    for i, bs in enumerate(batch_sizes):
        if i == 1:
            continue
        torch.manual_seed(100 + i)
        x_test = torch.randn(bs, 10)
        static_tensor_test = torch.randn(10)
        scale_test = 2
        multiplier_test = torch.randint(
            1, 10, (1,)
        ).item()  # 随机整数，验证 SymInt 动态性
        save_input_group(
            output_dir,
            "simple",
            i,
            x_test,
            static_tensor_test,
            scale_test,
            multiplier_test,
        )
        if i == 0:
            save_input_group(
                output_dir,
                "simple",
                i + 1,
                x_test,
                static_tensor_test,
                scale_test,
                multiplier_test,
            )

    # =========================================================
    # SimpleStaticModel (纯静态 shape)
    # =========================================================
    print("\n" + "=" * 60)
    print("SimpleStaticModel (static shape)")
    print("=" * 60)

    static_model = SimpleStaticModel().eval()

    # 静态输入（固定 shape，不使用 dynamic_shapes）
    x_static = torch.randn(4, 10)
    static_tensor_static = torch.randn(10)
    scale_static = 3
    multiplier_static = 5

    export_model(
        "model_static",
        static_model,
        (x_static, static_tensor_static, scale_static, multiplier_static),
        output_dir,
    )

    print("\nGenerating static shape input data...")
    # 3 组相同 shape 的输入（验证静态 shape 下多组数据精度）
    for i in range(3):
        torch.manual_seed(200 + i)
        x_test = torch.randn(4, 10)
        static_tensor_test = torch.randn(10)
        scale_test = 3
        multiplier_test = 5
        save_input_group(
            output_dir,
            "static",
            i,
            x_test,
            static_tensor_test,
            scale_test,
            multiplier_test,
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
