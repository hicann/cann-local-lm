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
    # A getitem feeding the graph output directly is replaced by a slice, so the
    # signature must follow; this pass does not validate, and a stale name would
    # only surface later as a confusing I/O contract failure.
    _resync_output_signature(exported_program)


def _resync_output_signature(exported_program: ExportedProgram) -> None:
    """Point output specs at the nodes the graph actually returns.

    ``replace_all_uses_with`` rewrites the ``output`` node's arguments but not
    ``graph_signature``, whose specs reference nodes by plain name. Replacing a
    node that feeds the graph output directly therefore leaves the signature
    naming a node that no longer exists. Mirrors what ExecuTorch's own
    ``exir/passes/reinplace.py`` does after it replaces nodes.
    """
    output_node = next(
        (
            node
            for node in exported_program.graph_module.graph.nodes
            if node.op == "output"
        ),
        None,
    )
    if output_node is None or not output_node.args:
        return
    returned = output_node.args[0]
    if not isinstance(returned, (tuple, list)):
        returned = (returned,)
    specs = exported_program.graph_signature.output_specs
    if len(specs) != len(returned):
        raise ValueError(
            f"output specs and graph outputs differ: {len(specs)} specs, "
            f"{len(returned)} graph outputs"
        )
    for spec, value in zip(specs, returned):
        name = getattr(spec.arg, "name", None)
        if isinstance(value, torch.fx.Node) and name is not None and name != value.name:
            spec.arg.name = value.name
