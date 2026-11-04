# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Lower ExecuTorch Edge targets back to ATen overloads."""

from __future__ import annotations

from torch.export import ExportedProgram


def lower_edge_ops_to_aten(exported_program: ExportedProgram) -> None:
    """Restore ATen overloads from newer ExecuTorch EdgeOpOverloads."""
    graph = exported_program.graph_module.graph
    changed = False
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        if not type(target).__module__.startswith("executorch.exir.dialects.edge"):
            continue
        aten_target = getattr(target, "_op", None)
        if aten_target is None:
            raise ValueError(
                f"cannot lower Edge target to ATen: node={node.name!r}, target={target!r}"
            )
        node.target = aten_target
        changed = True
    if changed:
        exported_program.graph_module.recompile()
