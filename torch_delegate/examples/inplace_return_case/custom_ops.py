#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Inplace+Return 测试 - 自定义算子定义

定义用于测试的自定义算子：
- inplace_scale_out: 一个 inplace 输入 + 返回 Tensor
    a *= scale (原地修改), 返回 a (修改后的张量)
"""

import torch
from torch.library import Library, impl

# 定义自定义算子库
my_ops = Library("my_ops", "DEF")

# inplace_scale_out: a *= scale (inplace), returns a (the modified tensor)
# 关键特性: 同时包含 inplace 输入 (a!) 和 Tensor 返回值
my_ops.define("inplace_scale_out(Tensor(a!) a, float scale) -> Tensor")


@impl(my_ops, "inplace_scale_out", "CPU")
def inplace_scale_out_cpu(a: torch.Tensor, scale: float) -> torch.Tensor:
    """CPU 实现"""
    a.mul_(scale)
    return a


@impl(my_ops, "inplace_scale_out", "PrivateUse1")
def inplace_scale_out_npu(a: torch.Tensor, scale: float) -> torch.Tensor:
    """NPU 实现"""
    a.mul_(scale)
    return a


@impl(my_ops, "inplace_scale_out", "Meta")
def inplace_scale_out_meta(a: torch.Tensor, scale: float) -> torch.Tensor:
    """Meta 实现（用于 shape 推导）"""
    return a
