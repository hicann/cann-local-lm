# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import torch
from executorch.exir.backend.canonical_partitioners.all_node_partitioner import (
    AllNodePartitioner,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export import ExportedProgram

# ExecuTorch discovers backend implementations from imported BackendDetails
# subclasses. Importing the public partitioner must therefore register the
# matching backend before lowering starts.
from .backend import AscendBackend as _AscendBackend  # noqa: F401
from .shared.constants import BACKEND_ID


class AscendPartitioner(AllNodePartitioner):
    """Full-graph partitioner for the torch-delegate bridge."""

    def __init__(self, compile_specs: list[CompileSpec]) -> None:
        """Assign the entire exported graph to Ascend with these CompileSpecs."""
        super().__init__(BACKEND_ID, compile_specs)

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> tuple[list[torch._ops.OpOverload], None]:
        """Preserve matmul, including nodes introduced by linear decomposition."""
        return [torch.ops.aten.matmul.default], None
