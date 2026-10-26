#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
E2E 测试 - 自定义算子定义

定义用于测试的自定义算子：
- scale_add: 支持 float 和 int 参数
"""

import torch
from torch.library import Library, impl

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
