# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import operator

import torch
from torch.export import ExportedProgram


def lower_static_split_with_sizes_copy(exported_program: ExportedProgram) -> None:
    """Rewrite indexed static splits as slices understood by torch-delegate.

    The rewrite is safe only when every split result is immediately selected
    by a constant ``getitem``; otherwise replacing the tuple-producing op would
    change observable graph behavior.
    """
    graph = exported_program.graph_module.graph
    for split_node in list(graph.nodes):
        if (
            split_node.op != "call_function"
            or split_node.target is not torch.ops.aten.split_with_sizes_copy.default
        ):
            continue
        split_args = split_node.args
        if len(split_args) < 2 or not isinstance(split_args[0], torch.fx.Node):
            raise ValueError("split_with_sizes_copy requires a tensor input and sizes")
        if not isinstance(split_args[1], (list, tuple)):
            raise ValueError("split_with_sizes_copy requires static split sizes")
        split_sizes = tuple(int(size) for size in split_args[1])
        dim = int(split_args[2]) if len(split_args) > 2 else 0
        users = list(split_node.users)
        if not users or any(
            user.op != "call_function" or user.target is not operator.getitem
            for user in users
        ):
            raise ValueError(
                "split_with_sizes_copy is supported only when all outputs are indexed"
            )
        for user in users:
            index = int(user.args[1])
            if index < 0:
                index += len(split_sizes)
            if index < 0 or index >= len(split_sizes):
                raise ValueError("split_with_sizes_copy getitem index is out of range")
            start = sum(split_sizes[:index])
            with graph.inserting_before(user):
                slice_node = graph.call_function(
                    torch.ops.aten.slice_copy.Tensor,
                    args=(split_args[0], dim, start, start + split_sizes[index]),
                )
            slice_node.meta = dict(user.meta)
            user.replace_all_uses_with(slice_node)
            graph.erase_node(user)
        graph.erase_node(split_node)
    graph.eliminate_dead_code()
    exported_program.graph_module.recompile()
