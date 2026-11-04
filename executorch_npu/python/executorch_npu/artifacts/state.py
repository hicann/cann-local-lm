# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Export-time state declarations and runtime state contract rules."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, Sequence, TYPE_CHECKING

import torch

from ..shared.diagnostics import log_state_mapping
from .manifest import IoContract, Manifest, StateBinding, StateContract


if TYPE_CHECKING:
    from torch.export import ExportedProgram


class StateSpec(Protocol):
    name: str
    dtype: str
    shape: tuple[int, ...]


def build_state_io_contract(
    exported_program: object,
    state_specs: Sequence[StateSpec],
    *,
    current_input_start: int,
    reset_owner: bool,
) -> dict[str, list[dict[str, object]]]:
    """Build the initial state contract from an exported graph signature."""
    signature = exported_program.graph_signature
    input_names = tuple(signature.user_inputs)
    output_names = tuple(signature.user_outputs)
    if len(input_names) != current_input_start + len(state_specs):
        raise ValueError("exported state input count does not match state specs")
    if len(output_names) != 1 + len(state_specs):
        raise ValueError("exported state output count does not match state specs")
    state_inputs = tuple(str(name) for name in input_names[current_input_start:])
    output_indices = range(1, len(output_names))
    return {
        "bindings": [
            {
                "state_id": spec.name,
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "reset_policy": "on_prefill",
                "reset_owner": reset_owner,
                "host_visibility": "synchronized",
                "next_output_arg": len(input_names) + output_index,
                "source_input_name": state_input,
                "source_output_arg": output_index,
            }
            for spec, state_input, output_index in zip(
                state_specs, state_inputs, output_indices, strict=True
            )
        ]
    }


def resolve_state_bindings(
    exported_program: ExportedProgram,
    raw_bindings: Any,
    *,
    input_args: Mapping[str, int],
    output_args: Mapping[str, int],
) -> list[StateBinding]:
    """Rebind export-time state identities to the delegated graph signature."""
    if not isinstance(raw_bindings, list):
        raise TypeError("state_io_contract.bindings must be an array")
    if not raw_bindings:
        return []
    if not all(isinstance(binding, Mapping) for binding in raw_bindings):
        raise TypeError("state_io_contract.bindings entries must be objects")
    bindings = [dict(binding) for binding in raw_bindings]

    nodes = {node.name: node for node in exported_program.graph_module.graph.nodes}
    input_names = list(input_args)
    output_names = list(output_args)
    state_input_names = {
        str(binding.get("source_input_name"))
        for binding in bindings
        if isinstance(binding.get("source_input_name"), str)
    }
    all_input_names = set(input_names)
    dependency_cache = {
        name: _graph_value_dependencies(name, nodes, all_input_names)
        for name in output_names
    }

    log_state_mapping(input_names, output_names, nodes, dependency_cache)

    used_output_names: set[str] = set()
    for binding in bindings:
        source_input_name = binding.get("source_input_name")
        if not isinstance(source_input_name, str):
            raise ValueError(
                "state binding lacks source graph-signature identity for state "
                f"{binding.get('state_id')!r}"
            )
        if source_input_name not in input_names:
            raise ValueError(
                "delegated graph is missing source state input "
                f"{source_input_name!r} for state {binding.get('state_id')!r}"
            )
        input_index = input_args[source_input_name]
        if not _matches_state_metadata(nodes.get(source_input_name), binding):
            raise ValueError(
                f"delegated state input metadata mismatch for {binding.get('state_id')!r}"
            )

        exclusive_matches = [
            output_name
            for output_name in output_names
            if output_name not in used_output_names
            and _matches_state_metadata(nodes.get(output_name), binding)
            and source_input_name in dependency_cache[output_name]
            and not (
                (dependency_cache[output_name] - {source_input_name})
                & state_input_names
            )
        ]
        if len(exclusive_matches) == 1:
            output_name = exclusive_matches[0]
        else:
            dependency_matches = [
                output_name
                for output_name in output_names
                if output_name not in used_output_names
                and _matches_state_metadata(nodes.get(output_name), binding)
                and source_input_name in dependency_cache[output_name]
            ]
            if dependency_matches:
                output_name = min(
                    dependency_matches,
                    key=lambda name: len(dependency_cache[name] & state_input_names),
                )
            else:
                state_is_used = any(
                    source_input_name in dependency_cache[output_name]
                    and _matches_state_metadata(nodes.get(output_name), binding)
                    for output_name in output_names
                )
                if state_is_used:
                    raise ValueError(
                        "could not map delegated state output for "
                        f"{binding.get('state_id')!r}: "
                        f"exclusive={exclusive_matches}, dep_candidates=[]"
                    )
                binding["current_input_arg"] = input_index
                binding["next_output_arg"] = -1
                binding.pop("source_input_name")
                binding.pop("source_output_arg", None)
                continue

        used_output_names.add(output_name)
        binding["current_input_arg"] = input_index
        binding["next_output_arg"] = output_args[output_name]
        binding.pop("source_input_name")
        binding.pop("source_output_arg", None)
    return [StateBinding(**binding) for binding in bindings]


def _graph_value_dependencies(
    value_name: str,
    nodes: Mapping[str, torch.fx.Node],
    input_names: set[str],
) -> set[str]:
    """Collect user inputs recursively consumed by one FX graph value."""
    dependencies: set[str] = set()
    visited: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, torch.fx.Node):
            if value.name in visited:
                return
            visited.add(value.name)
            if value.name in input_names:
                dependencies.add(value.name)
                return
            for argument in value.args:
                visit(argument)
            for argument in value.kwargs.values():
                visit(argument)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    node = nodes.get(value_name)
    if node is not None:
        visit(node)
    return dependencies


def _matches_state_metadata(
    node: torch.fx.Node | None, binding: Mapping[str, Any]
) -> bool:
    """Return whether a graph value matches the declared state dtype and shape."""
    tensor = None if node is None else node.meta.get("val")
    if not isinstance(tensor, torch.Tensor) and not hasattr(tensor, "dtype"):
        return False
    try:
        shape = [int(dimension) for dimension in tensor.shape]
    except (TypeError, ValueError):
        return False
    return str(tensor.dtype).removeprefix("torch.") == binding.get(
        "dtype"
    ) and shape == binding.get("shape")


def build_buffer_state_bindings(
    exported_program: ExportedProgram,
    buffer_slots: Sequence[tuple[str, str, int, int]],
) -> list[StateBinding]:
    """Describe mutated buffers using slots allocated by the I/O layout."""
    nodes = {node.name: node for node in exported_program.graph_module.graph.nodes}
    bindings = []
    for target, input_name, current_arg, next_arg in buffer_slots:
        tensor = nodes[input_name].meta.get("val")
        bindings.append(
            StateBinding(
                state_id=target,
                current_input_arg=current_arg,
                next_output_arg=next_arg,
                dtype=str(tensor.dtype).removeprefix("torch."),
                shape=[int(dimension) for dimension in tensor.shape],
                reset_policy="never",
                reset_owner=False,
                host_visibility="device_only",
            )
        )
    return bindings


def check_state_io(state: StateContract, io_contract: IoContract) -> None:
    """Validate each state against the resolved manifest argument bindings."""
    io_bindings = io_contract.inputs + io_contract.outputs
    for io in io_bindings:
        if io.semantic_role != "data" and io.shape_range is not None:
            raise ValueError("dynamic state tensors are not supported")
    io_by_arg = {binding.et_arg_index: binding for binding in io_bindings}
    state_ids: set[str] = set()
    state_args: set[int] = set()
    for binding in state.bindings:
        if binding.state_id in state_ids:
            raise ValueError(f"duplicate state_id: {binding.state_id}")
        state_ids.add(binding.state_id)
        _check_reset_policy(binding)
        for index, direction in (
            (binding.current_input_arg, "input"),
            (binding.next_output_arg, "output"),
        ):
            if index == -1:
                continue
            if index in state_args:
                raise ValueError(f"state argument {index} is reused")
            state_args.add(index)
            io = io_by_arg.get(index)
            if (
                io is None
                or io.direction != direction
                or io.dtype != binding.dtype
                or io.shape != binding.shape
            ):
                raise ValueError(f"state {binding.state_id!r} does not match graph I/O")


def _check_reset_policy(binding: StateBinding) -> None:
    if binding.reset_owner and binding.reset_policy != "on_prefill":
        raise ValueError("only on_prefill states may own reset")


def check_state_role(state: StateContract, role: str) -> None:
    """Keep reset ownership rules local to the state contract."""
    for binding in state.bindings:
        _check_reset_policy(binding)
        if binding.reset_owner and role != "prefill":
            raise ValueError("state reset owner must be a prefill delegate")


def check_shared_states(manifests: Iterable[Manifest]) -> None:
    """Check metadata agreement and reset ownership across delegate states."""
    state_metadata: dict[str, tuple[str, tuple[int, ...], str, str]] = {}
    reset_owners: dict[str, int] = {}
    for manifest in manifests:
        check_state_role(manifest.state, manifest.role)
        for binding in manifest.state.bindings:
            state_id = binding.state_id
            metadata = (
                binding.dtype,
                tuple(binding.shape),
                binding.reset_policy,
                binding.host_visibility,
            )
            prior = state_metadata.setdefault(state_id, metadata)
            if prior != metadata:
                raise ValueError(
                    f"state {state_id!r} metadata differs across delegates"
                )
            if binding.reset_owner:
                reset_owners[state_id] = reset_owners.get(state_id, 0) + 1
    for state_id, metadata in state_metadata.items():
        if metadata[2] == "on_prefill":
            owners = reset_owners.get(state_id, 0)
            if owners != 1:
                raise ValueError(
                    f"state {state_id!r} requires exactly one prefill reset owner, got {owners}"
                )
