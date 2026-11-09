# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import dataclasses
from abc import ABC, abstractmethod
from typing import Any

import torch


@dataclasses.dataclass
class NodeIR:
    """单个 EP 节点转换得到的 Ascend IR 描述。

    由 NodeIRConverter.convert_node() 产出，最终写入 EP JSON 中 node 对象的
    可选 ascend_ir 字段；source 为空字符串表示该节点走 fallback。
    """

    source: str
    ops: list[dict[str, Any]]  # GE opdef 列表
    mapping: dict[str, Any] | None = (
        None  # output 对应关系：aten_output_name -> ge_output_name
    )
    device: str = "npu"  # 执行 Device 归属："npu" | "cpu"


class NodeIRConverter(ABC):
    """EP 节点到 Ascend IR 的转换器抽象基类。

    实现类需保证 is_node_supported 与 convert_node 幂等且无副作用，
    convert_node 统一返回本模块定义的 NodeIR，返回 None 表示该节点不支持转换。
    当前唯一实现为 converter/torchair_converter.py 中的 TorchAirGEConverter。
    """

    @abstractmethod
    def is_node_supported(self, node: torch.fx.Node) -> bool: ...

    @abstractmethod
    def convert_node(self, node: torch.fx.Node) -> NodeIR | None: ...

    @abstractmethod
    def get_ascend_ir_map(self) -> dict[str, NodeIR]: ...
