# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Adapt ExecuTorch partitions through EPAIR to the CANN GE graph boundary."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from torch.utils._pytree import tree_unflatten

from ..shared.compilation import compile_lock
from ..shared.symbolic_shape import resolve_symbolic_dimension_range
from .compatibility import load_torch_delegate as _load_torch_delegate
from .compile_spec import AscendCompileOptions
from .offline import offline_compiler


def reconstruct_example_inputs(exported_program: ExportedProgram) -> None:
    """Recreate concrete examples required by torch-delegate re-export.

    ExecuTorch removes ``example_inputs`` from delegated ExportedPrograms, but
    torch-delegate still needs representative tensors when it re-exports the
    partition to EPAIR. Symbolic dimensions use their export-time optimization
    hints so finite tensors can be allocated without discarding the graph's
    symbolic constraints.
    """
    try:
        # PyTorch 2.12+ exposes the intent-specific spelling.
        from torch.fx.experimental.symbolic_shapes import optimization_hint
    except ImportError:
        # Older supported PyTorch releases expose the same behavior here.
        from torch.fx.experimental.symbolic_shapes import hint_int as optimization_hint

    if exported_program.example_inputs is not None:
        return
    nodes = {node.name: node for node in exported_program.graph_module.graph.nodes}
    flat_inputs: list[torch.Tensor] = []
    for spec in exported_program.graph_signature.input_specs:
        if spec.kind is not InputKind.USER_INPUT:
            continue
        node = nodes.get(spec.arg.name)
        value = None if node is None else node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "cannot reconstruct missing example input for "
                f"{spec.arg.name!r}: tensor metadata is unavailable"
            )
        concrete_shape = tuple(optimization_hint(dim) for dim in value.shape)
        flat_inputs.append(torch.zeros(concrete_shape, dtype=value.dtype))

    args, kwargs = tree_unflatten(flat_inputs, exported_program.call_spec.in_spec)
    exported_program._example_inputs = (args, kwargs)


@contextmanager
def _fallback_codegen_disabled() -> Iterator[None]:
    """Disable torch-delegate fallback codegen while extracting the GE graph.

    Fallback nodes cannot be represented in the standalone Ascend artifact, so
    compilation must fail instead of silently emitting host fallback code. The
    caller's prior environment value is restored even when extraction fails.
    """
    with compile_lock:
        name = "TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN"
        previous = os.environ.get(name)
        os.environ[name] = "0"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


def validate_aten_targets(
    exported_program: ExportedProgram, method_name: str, delegate_index: int
) -> None:
    """Reject Edge dialect targets before passing the graph to torch-delegate.

    The bridge currently consumes ATen overloads directly; accepting Edge
    dialect nodes would defer the failure until a less actionable conversion
    or compilation error.
    """
    for node in exported_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        if (
            "executorch_exir_dialects_edge" in target
            or target.startswith("edge.")
            or "::edge::" in target
        ):
            raise ValueError(
                "Ascend delegation requires ATen targets; "
                f"method={method_name!r}, delegate_index={delegate_index}, "
                f"node={node.name!r}, target={target!r}. Use "
                "EdgeCompileConfig(_use_edge_ops=False)."
            )


def _delegate_runtime_input_names(
    exported_program: ExportedProgram,
) -> list[str]:
    """Return GE Data/RefData inputs in graph-signature order."""
    signature = exported_program.graph_signature
    mutated_buffers = set(signature.buffers_to_mutate.values())
    names: list[str] = []
    for spec in signature.input_specs:
        if spec.kind is InputKind.USER_INPUT or (
            spec.kind is InputKind.BUFFER and spec.target in mutated_buffers
        ):
            name = getattr(spec.arg, "name", None)
            if isinstance(name, str):
                names.append(name)
    return names


def dynamic_input_shape_option(exported_program: ExportedProgram) -> str | None:
    """Build CANN's index-ordered ``input_shape`` option when inputs vary."""
    nodes = {node.name: node for node in exported_program.graph_module.graph.nodes}
    encoded_inputs: list[str] = []
    dynamic = False
    for raw_name in _delegate_runtime_input_names(exported_program):
        node = nodes.get(str(raw_name))
        tensor = None if node is None else node.meta.get("val")
        if tensor is None or not hasattr(tensor, "shape"):
            raise TypeError(
                f"cannot build CANN input_shape for {raw_name!r}: tensor metadata is unavailable"
            )
        dimensions: list[str] = []
        for dimension in tensor.shape:
            if isinstance(dimension, int):
                dimensions.append(str(int(dimension)))
                continue
            lower, upper = resolve_symbolic_dimension_range(exported_program, dimension)
            dimensions.append(f"{lower}~{upper}")
            dynamic = True
        encoded_inputs.append(",".join(dimensions))
    return ";".join(encoded_inputs) if dynamic else None


def _promote_mutated_buffers_to_runtime_inputs(contents: object) -> None:
    """Keep mutated buffers as GE runtime inputs instead of model constants.

    ExecuTorch owns registered buffers, while a whole-graph CANN delegate must
    keep them in persistent device state. TorchDelegate otherwise lowers every
    ``InputKind.BUFFER`` placeholder to a Const/FileConstant, which makes the
    functionalized buffer-mutation outputs impossible to feed into the next
    invocation.
    """
    ep_json = getattr(contents, "ep_json", None)
    if not isinstance(ep_json, dict):
        return
    signature = ep_json.get("graph_module", {}).get("signature", {})
    input_specs = signature.get("input_specs", [])
    output_specs = signature.get("output_specs", [])
    mutated_buffers = {
        spec["buffer_mutation"]["buffer_name"]
        for spec in output_specs
        if isinstance(spec, dict) and "buffer_mutation" in spec
    }
    for spec in input_specs:
        if not isinstance(spec, dict) or "buffer" not in spec:
            continue
        buffer_spec = spec["buffer"]
        if buffer_spec.get("buffer_name") not in mutated_buffers:
            continue
        placeholder_name = buffer_spec["arg"]["name"]
        spec.clear()
        spec["user_input"] = {"arg": {"as_tensor": {"name": placeholder_name}}}


def _initialize_ge(soc: str) -> None:
    """Share GE initialization while letting torch-delegate configure TorchAir."""
    # The converter initializes TorchAir with its full compilation options.
    # An earlier parameterless call would freeze an empty option dictionary.
    offline_compiler.initialize(soc)


def lower_to_ge_graph(
    exported_program: ExportedProgram,
    specs: AscendCompileOptions,
    epair_path: Path,
) -> Any:
    """Return an independently owned GE graph, closing its EPAIR loader before return."""
    with compile_lock:
        validate_aten_targets(exported_program, specs.method_name, specs.delegate_index)
        reconstruct_example_inputs(exported_program)
        torch_delegate = _load_torch_delegate(specs.acl_soc_name)
        _initialize_ge(specs.acl_soc_name)
        contents = torch_delegate.export_to_epair(exported_program)
        if contents.manifest.converted_nodes != contents.manifest.total_nodes:
            raise RuntimeError(
                "torch-delegate did not fully convert the delegated graph: "
                f"method={specs.method_name!r}, "
                f"delegate_index={specs.delegate_index}, "
                f"converted={contents.manifest.converted_nodes}, "
                f"total={contents.manifest.total_nodes}"
            )
        _promote_mutated_buffers_to_runtime_inputs(contents)
        torch_delegate.epair_save(contents, epair_path)
        with (
            _fallback_codegen_disabled(),
            torch_delegate.EpairModelLoader(epair_path) as loader,
        ):
            return loader.get_ge_graph()
