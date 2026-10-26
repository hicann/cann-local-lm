#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Fallback Inplace/Optional 测试 - 自定义算子定义

定义用于测试的自定义算子：
- dual_inplace_: 多 inplace 参数（两个 tensor 都被原地修改）
- multi_optional_op: 多可选参数（4 个可选参数，测试部分传入）
"""

import torch
from torch.library import Library, impl

# 定义自定义算子库
my_ops = Library("my_ops", "DEF")

# dual_inplace_: a *= scale_a; b *= scale_b
# 两个 tensor 参数都是 inplace 的
my_ops.define(
    "dual_inplace_(Tensor(a!) a, Tensor(b!) b, float scale_a, float scale_b) -> ()"
)


@impl(my_ops, "dual_inplace_", "CPU")
def dual_inplace_cpu(
    a: torch.Tensor, b: torch.Tensor, scale_a: float, scale_b: float
) -> None:
    """CPU 实现"""
    a.mul_(scale_a)
    b.mul_(scale_b)


@impl(my_ops, "dual_inplace_", "PrivateUse1")
def dual_inplace_npu(
    a: torch.Tensor, b: torch.Tensor, scale_a: float, scale_b: float
) -> None:
    """NPU 实现"""
    a.mul_(scale_a)
    b.mul_(scale_b)


@impl(my_ops, "dual_inplace_", "Meta")
def dual_inplace_meta(
    a: torch.Tensor, b: torch.Tensor, scale_a: float, scale_b: float
) -> None:
    """Meta 实现（用于 shape 推导）"""
    pass


# multi_optional_op: result = x * alpha + beta + gamma + delta
# 4 个可选参数，都有默认值
my_ops.define(
    "multi_optional_op(Tensor x, "
    "float alpha=1.0, "
    "float beta=0.0, "
    "float gamma=0.0, "
    "float delta=0.0) -> Tensor"
)


@impl(my_ops, "multi_optional_op", "CPU")
def multi_optional_op_cpu(
    x: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    delta: float = 0.0,
) -> torch.Tensor:
    """CPU 实现"""
    return x * alpha + beta + gamma + delta


@impl(my_ops, "multi_optional_op", "PrivateUse1")
def multi_optional_op_npu(
    x: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    delta: float = 0.0,
) -> torch.Tensor:
    """NPU 实现"""
    return x * alpha + beta + gamma + delta


@impl(my_ops, "multi_optional_op", "Meta")
def multi_optional_op_meta(
    x: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 0.0,
    delta: float = 0.0,
) -> torch.Tensor:
    """Meta 实现（用于 shape 推导）"""
    return torch.empty_like(x)
