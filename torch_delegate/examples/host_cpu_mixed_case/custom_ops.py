#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
HostCpu 混合调度示例 - 自定义算子定义

只注册 CPU 实现（无 PrivateUse1 kernel），Device 归属自动判定会把这些算子
判为 device="cpu"，由 Fallback Codegen 生成注册到 GE HostCpu 后端的自定义算子：
- hcpu_ops::feature_scale: Tensor x Scalar（单输入 + 标量参数）
- hcpu_ops::feature_merge: Tensor x Tensor（双输入）
"""

import torch
from torch.library import Library, impl

hcpu_ops = Library("hcpu_ops", "DEF")

# feature_scale: CPU-only 算子，输出 shape = 输入 shape
hcpu_ops.define("feature_scale(Tensor self, Scalar factor) -> Tensor")


@impl(hcpu_ops, "feature_scale", "CPU")
def feature_scale_cpu(self_, factor):
    return self_ * float(factor)


@impl(hcpu_ops, "feature_scale", "Meta")
def feature_scale_meta(self_, factor):
    return torch.empty_like(self_)


# feature_merge: CPU-only 算子，双输入逐元素融合
hcpu_ops.define("feature_merge(Tensor a, Tensor b) -> Tensor")


@impl(hcpu_ops, "feature_merge", "CPU")
def feature_merge_cpu(a, b):
    return a + b


@impl(hcpu_ops, "feature_merge", "Meta")
def feature_merge_meta(a, b):
    return torch.empty_like(a)


# feature_gather: CPU-only embedding 查表（推荐网络 CPU 前处理段）
# 规避说明：ids 以 float32 传入、kernel 内转回 int64 查表。GE runtime 对 HostCpu
# 算子的 Data（用户运行时输入）做 device→host staging 时按 4 字节/元素计算拷贝
# 长度，DT_INT64 输入会被截断（构图侧 fail-fast 检测见
# FallbackNodePlanner::BuildInputDefs）；float32 输入不受影响。GE 修复后可还原。
hcpu_ops.define("feature_gather(Tensor table, Tensor ids_f) -> Tensor")


@impl(hcpu_ops, "feature_gather", "CPU")
def feature_gather_cpu(table, ids_f):
    return table.index_select(0, ids_f.to(torch.long))


@impl(hcpu_ops, "feature_gather", "Meta")
def feature_gather_meta(table, ids_f):
    return torch.empty(
        (ids_f.size(0), table.size(1)), dtype=table.dtype, device=table.device
    )


# score_rank: CPU-only 排序打分偏置（推荐网络 CPU 后处理段）
hcpu_ops.define("score_rank(Tensor scores, Tensor bias) -> Tensor")


@impl(hcpu_ops, "score_rank", "CPU")
def score_rank_cpu(scores, bias):
    return scores + bias


@impl(hcpu_ops, "score_rank", "Meta")
def score_rank_meta(scores, bias):
    return torch.empty_like(scores)
