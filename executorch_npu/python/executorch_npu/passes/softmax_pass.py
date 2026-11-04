# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Prepare ATen softmax for the Ascend GE converter before Edge lowering."""

from __future__ import annotations

import copy

import torch
from torch.export import ExportedProgram


def lower_softmax_int(exported_program: ExportedProgram) -> int:
    """Replace softmax.int with a dtype cast and GE-supported _softmax."""
    graph = exported_program.graph_module.graph
    target = torch.ops.aten.softmax.int
    replaced = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != target:
            continue

        source = node.args[0]
        dim = node.args[1]
        dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
        value = source.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"softmax input metadata missing for {node.name}")

        with graph.inserting_before(node):
            if dtype is not None and value.dtype != dtype:
                cast = graph.call_function(
                    torch.ops.aten._to_copy.default, (source,), {"dtype": dtype}
                )
                cast.meta = copy.copy(source.meta)
                cast.meta["val"] = value.to(dtype=dtype)
                tensor_meta = cast.meta.get("tensor_meta")
                if tensor_meta is not None and hasattr(tensor_meta, "_replace"):
                    cast.meta["tensor_meta"] = tensor_meta._replace(dtype=dtype)
                source = cast

            replacement = graph.call_function(
                torch.ops.aten._softmax.default, (source, dim, False)
            )
            replacement.meta = copy.copy(node.meta)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        replaced += 1

    if replaced:
        graph.lint()
        exported_program.graph_module.recompile()
        exported_program.validate()
    return replaced
