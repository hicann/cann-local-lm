# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import torch
from torch.export import ExportedProgram


def lower_edge_copy_ops(exported_program: ExportedProgram) -> None:
    """Lower unsupported copy overloads without exposing view aliases.

    torch-delegate accepts the corresponding view operators. A following
    ``clone`` restores each ``*_copy`` operator's non-aliasing contract for
    downstream users.
    """
    targets = {
        torch.ops.aten.alias_copy.default: torch.ops.aten.alias.default,
        torch.ops.aten.expand_copy.default: torch.ops.aten.expand.default,
        torch.ops.aten.permute_copy.default: torch.ops.aten.permute.default,
        torch.ops.aten.select_copy.int: torch.ops.aten.select.int,
        torch.ops.aten.slice_copy.Tensor: torch.ops.aten.slice.Tensor,
        torch.ops.aten.unsqueeze_copy.default: torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view_copy.default: torch.ops.aten.view.default,
    }
    graph = exported_program.graph_module.graph
    for node in list(graph.nodes):
        if _is_dim_order_clone(node):
            dim_order = node.kwargs.get("dim_order")
            value = node.meta.get("val")
            identity_dim_order = (
                list(range(value.ndim)) if isinstance(value, torch.Tensor) else None
            )
            if dim_order != identity_dim_order:
                raise ValueError(
                    f"_clone_dim_order lowering requires an identity dim_order, got {dim_order!r}"
                )
            node.target = torch.ops.aten.clone.default
            node.kwargs = {}
            continue
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten.squeeze_copy.dims
        ):
            value = node.meta.get("val")
            source = node.args[0]
            source_value = (
                source.meta.get("val") if isinstance(source, torch.fx.Node) else None
            )
            if not isinstance(value, torch.Tensor) or not isinstance(
                source_value, torch.Tensor
            ):
                raise TypeError("squeeze_copy is missing tensor metadata")
            if not source_value.is_contiguous():
                raise ValueError(
                    "squeeze_copy lowering requires a contiguous input tensor"
                )
            node.target = torch.ops.aten.view.default
            node.args = (node.args[0], list(value.shape))
            node.kwargs = {}
            users = list(node.users)
            with graph.inserting_after(node):
                clone = graph.call_function(torch.ops.aten.clone.default, (node,))
            clone.meta = dict(node.meta)
            for user in users:
                user.replace_input_with(node, clone)
            continue
        if node.op == "call_function" and node.target in targets:
            node.target = targets[node.target]
            users = list(node.users)
            with graph.inserting_after(node):
                clone = graph.call_function(torch.ops.aten.clone.default, (node,))
            clone.meta = dict(node.meta)
            for user in users:
                user.replace_input_with(node, clone)
    exported_program.graph_module.recompile()


def _is_dim_order_clone(node: torch.fx.Node) -> bool:
    schema = getattr(node.target, "_schema", None)
    return node.op == "call_function" and getattr(schema, "name", None) == (
        "dim_order_ops::_clone_dim_order"
    )
