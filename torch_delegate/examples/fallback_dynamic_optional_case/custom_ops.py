#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Fallback Dynamic/Optional 测试 - 自定义算子定义

定义用于测试的自定义算子：
- my_addn: 动态输入 Tensor[] (对应 GE DYNAMIC_INPUT)
- my_addmm: 可选输入 Tensor? (对应 GE OPTIONAL_INPUT) + Scalar 参数
"""

import torch
from torch.library import Library, impl

my_ops = Library("my_ops", "DEF")

# my_addn: 动态输入 — tensors 是 Tensor[]，对应 GE DYNAMIC_INPUT
# 输出 shape = 输入 shape（逐元素求和），InferShape 简单
# IR layout: [0] DYNAMIC_INPUT(tensors)
my_ops.define("my_addn(Tensor[] tensors) -> Tensor")


@impl(my_ops, "my_addn", "CPU")
def my_addn_cpu(tensors):
    result = tensors[0].clone()
    for t in tensors[1:]:
        result = result + t
    return result


@impl(my_ops, "my_addn", "PrivateUse1")
def my_addn_npu(tensors):
    result = tensors[0].clone()
    for t in tensors[1:]:
        result = result + t
    return result


@impl(my_ops, "my_addn", "Meta")
def my_addn_meta(tensors):
    return torch.empty_like(tensors[0])


# my_addmm: 可选输入 — bias 是 Tensor?，调用时传 None 不占位
# IR layout: [0] INPUT(input), [1] INPUT(mat1), [2] INPUT(mat2),
#            [3] OPTIONAL_INPUT(bias), [4] INPUT(beta)
my_ops.define(
    "my_addmm(Tensor input, Tensor mat1, Tensor mat2, Tensor? bias, *, Scalar beta=1) -> Tensor"
)


@impl(my_ops, "my_addmm", "CPU")
def my_addmm_cpu(input, mat1, mat2, bias=None, *, beta=1):
    result = input + beta * torch.mm(mat1, mat2)
    if bias is not None:
        result = result + bias
    return result


@impl(my_ops, "my_addmm", "PrivateUse1")
def my_addmm_npu(input, mat1, mat2, bias=None, *, beta=1):
    result = input + beta * torch.mm(mat1, mat2)
    if bias is not None:
        result = result + bias
    return result


@impl(my_ops, "my_addmm", "Meta")
def my_addmm_meta(input, mat1, mat2, bias=None, *, beta=1):
    return torch.empty(mat1.shape[0], mat2.shape[1], device="meta")
