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
    source: str
    ops: list[dict[str, Any]]  # GE opdef 列表
    mapping: dict[str, Any] | None = (
        None  # output 对应关系：aten_output_name -> ge_output_name
    )


class NodeIRConverter(ABC):
    @abstractmethod
    def is_node_supported(self, node: torch.fx.Node) -> bool: ...

    @abstractmethod
    def convert_node(self, node: torch.fx.Node) -> NodeIR | None: ...

    @abstractmethod
    def get_ascend_ir_map(self) -> dict[str, NodeIR]: ...
