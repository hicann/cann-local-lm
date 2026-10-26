# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Inplace 模型定义

定义三种带有 inplace 操作的模型，供所有测试脚本跨进程导入使用。
"""

import torch


class SimpleInplaceModel(torch.nn.Module):
    """简单 inplace add 模型

    forward(x, y) -> x.add_(y) -> return x
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x.add_(y)  # inplace add
        x.mul_(self.scale)  # inplace mul
        return x


class ChainInplaceModel(torch.nn.Module):
    """链式 inplace 操作模型

    forward(x) -> x.add_(1) -> x.mul_(2) -> x.div_(4) -> return x
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)  # x = x + 1
        x.mul_(2.0)  # x = x * 2
        x.div_(4.0)  # x = x / 4
        return x


class ParamInplaceModel(torch.nn.Module):
    """带参数的 inplace 模型

    forward(x) -> x.add_(bias) -> x.mul_(scale) -> return x
    """

    def __init__(self, scale: float = 2.0):
        super().__init__()
        self.scale = scale
        self.register_buffer("bias", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(self.bias)  # inplace add with buffer
        x.mul_(self.scale)  # inplace mul with parameter
        return x


# 模型配置表：名称 -> (模型类, 构造参数, 输入形状列表)
MODEL_CONFIGS = {
    "simple": (SimpleInplaceModel, (2.0,), [(4, 4), (4, 4)]),
    "chain": (ChainInplaceModel, (), [(4, 4)]),
    "param": (ParamInplaceModel, (2.0,), [(4, 4)]),
}


def get_model(name: str, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """根据名称获取模型实例"""
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    model_class, model_args, _ = MODEL_CONFIGS[name]
    model = model_class(*model_args).to(device=device, dtype=dtype).eval()
    return model


def get_input_shapes(name: str) -> list:
    """根据模型名称获取输入形状"""
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    _, _, input_shapes = MODEL_CONFIGS[name]
    return input_shapes
