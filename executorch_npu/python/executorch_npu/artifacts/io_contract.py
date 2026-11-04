# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Construct manifest I/O bindings from the delegated graph signature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind

from ..shared.symbolic_shape import resolve_symbolic_dimension_range
from .manifest import Direction, IoBinding, IoContract, SemanticRole, StateContract
from .state import build_buffer_state_bindings, check_state_io, resolve_state_bindings


@dataclass
class _ArgumentLayout:
    """One source of EValue indices for I/O construction and state resolution."""

    inputs: list[tuple[str, int]]
    outputs: list[tuple[str, int]]
    user_inputs: dict[str, int]
    user_outputs: dict[str, int]
    buffer_states: list[tuple[str, str, int, int]]


def _tensor_argument_name(spec: object) -> str:
    name = getattr(getattr(spec, "arg", None), "name", None)
    if not isinstance(name, str):
        raise TypeError("CANN delegate I/O must be tensor arguments")
    return name


def _argument_layout(exported_program: ExportedProgram) -> _ArgumentLayout:
    """Preserve GE boundary order while assigning ExecuTorch argument slots."""
    signature = exported_program.graph_signature
    input_names = [str(name) for name in signature.user_inputs]
    output_names = [str(name) for name in signature.user_outputs]
    inputs = [(name, index) for index, name in enumerate(input_names)]
    outputs = [
        (name, len(input_names) + index) for index, name in enumerate(output_names)
    ]
    user_inputs: dict[str, int] = {}
    user_outputs: dict[str, int] = {}
    for name, index in inputs:
        user_inputs.setdefault(name, index)
    for name, index in outputs:
        user_outputs.setdefault(name, index)
    layout = _ArgumentLayout(inputs, outputs, user_inputs, user_outputs, [])

    mutated_targets = set(signature.buffers_to_mutate.values())
    if not mutated_targets:
        return layout

    ordered_targets = [
        spec.target
        for spec in signature.input_specs
        if spec.kind is InputKind.BUFFER and spec.target in mutated_targets
    ]
    mutation_specs = [
        spec
        for spec in signature.output_specs
        if spec.kind is OutputKind.BUFFER_MUTATION
    ]
    if len(mutation_specs) != len(ordered_targets):
        raise ValueError("mutated buffer input/output counts differ")
    mutation_by_target = {
        signature.buffers_to_mutate[_tensor_argument_name(spec)]: spec
        for spec in mutation_specs
    }
    if set(mutation_by_target) != set(ordered_targets):
        raise ValueError("mutated buffer input/output identities differ")
    input_by_target = {
        spec.target: _tensor_argument_name(spec)
        for spec in signature.input_specs
        if spec.kind is InputKind.BUFFER and spec.target in mutated_targets
    }
    user_arg_count = len(input_names) + len(output_names)
    current_args: dict[str, int] = {}
    layout.inputs = []
    layout.outputs = []
    for index, target in enumerate(ordered_targets):
        current_arg = user_arg_count + 2 * index
        next_arg = current_arg + 1
        current_args[target] = current_arg
        layout.buffer_states.append(
            (target, input_by_target[target], current_arg, next_arg)
        )
        layout.outputs.append(
            (_tensor_argument_name(mutation_by_target[target]), next_arg)
        )
    for spec in signature.input_specs:
        if spec.kind is InputKind.BUFFER and spec.target in current_args:
            layout.inputs.append(
                (_tensor_argument_name(spec), current_args[spec.target])
            )
        elif spec.kind is InputKind.USER_INPUT:
            name = _tensor_argument_name(spec)
            layout.inputs.append((name, user_inputs[name]))
    for spec in signature.output_specs:
        if spec.kind is OutputKind.USER_OUTPUT:
            name = _tensor_argument_name(spec)
            layout.outputs.append((name, user_outputs[name]))
    return layout


def build_io_contract(
    exported_program: ExportedProgram,
    state_io_contract: Mapping[str, Any] | None = None,
) -> tuple[IoContract, StateContract]:
    """Build typed I/O and state contracts against one argument layout."""
    layout = _argument_layout(exported_program)
    raw_bindings = (state_io_contract or {}).get("bindings", [])
    bindings = resolve_state_bindings(
        exported_program,
        raw_bindings,
        input_args=layout.user_inputs,
        output_args=layout.user_outputs,
    )
    if layout.buffer_states:
        if bindings:
            raise ValueError(
                "explicit state contracts cannot be combined with mutated module buffers"
            )
        bindings = build_buffer_state_bindings(exported_program, layout.buffer_states)
    state = StateContract(bindings=bindings)
    roles: dict[tuple[Direction, int], SemanticRole] = {}
    for binding in bindings:
        roles[("input", binding.current_input_arg)] = "state_current"
        if binding.next_output_arg >= 0:
            roles[("output", binding.next_output_arg)] = "state_next"

    nodes = {node.name: node for node in exported_program.graph_module.graph.nodes}
    io = IoContract(
        inputs=[
            _make_binding(exported_program, nodes, name, index, "input", roles)
            for name, index in layout.inputs
        ],
        outputs=[
            _make_binding(exported_program, nodes, name, index, "output", roles)
            for name, index in layout.outputs
        ],
    )
    check_state_io(state, io)
    return io, state


def _make_binding(
    exported_program: ExportedProgram,
    nodes: Mapping[str, torch.fx.Node],
    node_name: str,
    argument_index: int,
    direction: Direction,
    roles: Mapping[tuple[Direction, int], SemanticRole],
) -> IoBinding:
    """Translate one graph tensor into the manifest's I/O representation."""
    node = nodes.get(node_name)
    tensor = None if node is None else node.meta.get("val")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"graph argument {node_name!r} is not a tensor")
    return IoBinding(
        et_arg_index=argument_index,
        direction=direction,
        ascend_name=node_name,
        dtype=str(tensor.dtype).removeprefix("torch."),
        semantic_role=roles.get((direction, argument_index), "data"),
        **tensor_shape_contract(exported_program, tensor),
    )


def tensor_shape_contract(
    exported_program: ExportedProgram, tensor: torch.Tensor
) -> dict[str, list[Any]]:
    """Return canonical static/dynamic manifest metadata for one tensor."""
    shape: list[int] = []
    shape_range: list[list[int]] = []
    dynamic = False
    for dimension in tensor.shape:
        if isinstance(dimension, int):
            value = int(dimension)
            shape.append(value)
            shape_range.append([value, value])
            continue
        lower, upper = resolve_symbolic_dimension_range(exported_program, dimension)
        shape.append(-1)
        shape_range.append([lower, upper])
        dynamic = True
    result: dict[str, list[Any]] = {"shape": shape}
    if dynamic:
        result["shape_range"] = shape_range
    return result
