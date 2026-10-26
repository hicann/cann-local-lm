# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import dataclasses
import torch
from typing import Any


@dataclasses.dataclass
class EpairConfig:
    """Epair 转换配置。

    控制 ExportedProgram 到 Ascend IR 的转换行为。

    Attributes:
        remove_noop_ops: 是否移除无操作节点（如 identity、reshape 等）。
            默认为 True。移除后可简化图结构，但可能影响调试。
        pattern_fusion_pass: 是否启用模式融合优化。
            默认为 True。启用后会将多个算子融合为单个 Ascend 算子，
            提升 NPU 执行效率。
        whitelist_decompositions: 允许分解的算子集合。
            默认包含以下算子：
            - torch.ops.aten.linear.default
            - torch.ops.aten.matmul.default
            - torch.ops.aten.layer_norm.default
            - torch.ops.aten.softmax.int
            - torch.ops.aten.log_softmax.int
            - torch.ops.aten.__and__.Tensor
            - torch.ops.aten.__or__.Tensor
            - torch.ops.aten.isfinite.default
            - torch.ops.aten.negative.default
            - torch.ops.aten.absolute.default
            - torch.ops.aten.relu6.default
            - torch.ops.aten.arccos.default
            - torch.ops.aten.arcsin.default
            - torch.ops.aten.arctan.default
            - torch.ops.aten.arccosh.default
            - torch.ops.aten.arcsinh.default
            - torch.ops.aten.conv2d.default
            - torch.ops.aten.square.default

            这些算子会被分解为更基础的算子，以便 Ascend 后端处理。
            设置为空 set() 可禁用所有分解。

    Example:
        >>> from torch_delegate import export_to_epair, EpairConfig
        >>> # 禁用模式融合（用于调试）
        >>> config = EpairConfig(pattern_fusion_pass=False)
        >>> contents = export_to_epair(ep, config)
        >>>
        >>> # 禁用所有算子分解
        >>> config = EpairConfig(whitelist_decompositions=set())
        >>> contents = export_to_epair(ep, config)
    """

    remove_noop_ops: bool = True
    pattern_fusion_pass: bool = True

    whitelist_decompositions: set[Any] = dataclasses.field(
        default_factory=lambda: {
            torch.ops.aten.linear.default,
            torch.ops.aten.matmul.default,
            torch.ops.aten.layer_norm.default,
            torch.ops.aten.softmax.int,
            torch.ops.aten.log_softmax.int,
            torch.ops.aten.__and__.Tensor,
            torch.ops.aten.__or__.Tensor,
            torch.ops.aten.isfinite.default,
            torch.ops.aten.negative.default,
            torch.ops.aten.absolute.default,
            torch.ops.aten.relu6.default,
            torch.ops.aten.arccos.default,
            torch.ops.aten.arcsin.default,
            torch.ops.aten.arctan.default,
            torch.ops.aten.arccosh.default,
            torch.ops.aten.arcsinh.default,
            torch.ops.aten.conv2d.default,
            torch.ops.aten.square.default,
        }
    )
