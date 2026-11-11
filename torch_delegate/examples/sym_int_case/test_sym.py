# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os

import torch
import torch.nn as nn
from torch.export import Dim

from torch.library import Library, impl

import torch_delegate as td

# 定义自定义算子库
my_ops = Library("my_ops", "DEF")

# scale_add: y = x * scale + shift
my_ops.define("scale_add(Tensor x, Tensor y, float scale, SymInt shift=0) -> Tensor")


@impl(my_ops, "scale_add", "CPU")
def scale_add_cpu(
    x: torch.Tensor, y: torch.Tensor, scale: float, shift: torch.SymInt = 0
) -> torch.Tensor:
    """CPU 实现"""
    return x * scale + y + shift


@impl(my_ops, "scale_add", "PrivateUse1")
def scale_add_npu(
    x: torch.Tensor, y: torch.Tensor, scale: float, shift: torch.SymInt = 0
) -> torch.Tensor:
    """NPU 实现（使用 torch 算子）"""
    return x * scale + y + shift


@impl(my_ops, "scale_add", "Meta")
def scale_add_meta(
    x: torch.Tensor, y: torch.Tensor, scale: float, shift: torch.SymInt = 0
) -> torch.Tensor:
    """Meta 实现（用于 shape 推导）"""
    return torch.empty_like(x)


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
        y = torch.sigmoid(x)
        y = torch.tanh(y)

        y = y + static_tensor

        y = torch.ops.my_ops.scale_add(y, y, 0.5, scale)
        # y = y * 0.5 + y + scale

        sym_multiplier = torch.sym_int(multiplier)
        y = torch.ops.my_ops.scale_add(y, y, 1.0, sym_multiplier)
        # y = y * 1.0 + y + sym_multiplier

        return y


if __name__ == "__main__":
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

    example_inputs = (x, static_tensor, scale, multiplier)
    ep = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    here = os.path.dirname(os.path.abspath(__file__))
    torch.export.save(ep, os.path.join(here, "test_sym.pt2"))
    ep_load = torch.export.load(os.path.join(here, "test_sym.pt2"))
    print("----------------------------------------------------")
    print(ep_load.module()(x, static_tensor, scale, multiplier))
    print("----------------------------------------------------")
    print(ep_load.module()(x, static_tensor, scale, 8))
    print("----------------------------------------------------")

    epair_content = td.export_to_epair(ep)
    td.epair_save(epair_content, os.path.join(here, "test_sym.epair"))
