# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile and package one Ascend partition within a staging scope."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path

from executorch.exir._serialize._named_data_store import NamedDataStore
from torch.export import ExportedProgram

from ..artifacts.builder import build_manifest
from ..artifacts.io_contract import build_io_contract
from ..artifacts.manifest import Manifest
from ..passes import (
    fold_scalar_constant_casts,
    lower_edge_copy_ops,
    lower_edge_ops_to_aten,
    lower_static_split_with_sizes_copy,
)
from ..shared.compilation import compile_lock
from ..shared.diagnostics import diagnostic, external_call, log_memory
from .compile_spec import AscendCompileOptions
from .delegate_adapter import (
    dynamic_input_shape_option,
    lower_to_ge_graph,
    reject_delegated_graph,
)
from .offline import offline_compiler


def compile_partition(
    edge_program: ExportedProgram,
    specs: AscendCompileOptions,
    store: NamedDataStore,
    staging_root: Path,
) -> Manifest:
    """Compile a graph and consume its files before the staging scope closes."""
    with compile_lock:
        # Reclaim the previous method's transient allocations (Python cycles +
        # glibc arena residue) before this method's GE phase ramps up.
        import ctypes
        import gc

        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        identity = f"{_path_part(specs.method_name)}_d{specs.delegate_index}"
        partition_dir = staging_root / identity
        partition_dir.mkdir(parents=True, exist_ok=False)
        epair_path = partition_dir / "model.epair"
        output_base = partition_dir / "model"
        weight_dir = partition_dir / "weight"
        weight_dir.mkdir()
        diagnostic(
            "compile_partition_start",
            method=specs.method_name,
            delegate_index=specs.delegate_index,
        )
        log_memory(f"ENTRY method={specs.method_name} d{specs.delegate_index}")
        # Reject an already-lowered graph before any pass rewrites it, so the
        # caller sees the misuse instead of a downstream fx2ge conversion error.
        reject_delegated_graph(edge_program, specs.method_name, specs.delegate_index)
        lower_edge_ops_to_aten(edge_program)
        lower_static_split_with_sizes_copy(edge_program)
        lower_edge_copy_ops(edge_program)
        folded_casts = fold_scalar_constant_casts(edge_program)
        diagnostic("scalar_constant_casts_folded", count=folded_casts)
        # Reject state contract mismatches before expensive GE lowering/compilation.
        build_io_contract(edge_program, specs.state_io_contract)

        build_options = dict(specs.offline_build_options)
        input_shape = dynamic_input_shape_option(edge_program)
        if input_shape is not None:
            build_options["input_shape"] = input_shape

        graph = lower_to_ge_graph(edge_program, specs, epair_path)
        # The closed EPAIR loader returned an independently owned GE graph.
        # Release the serialized weights before GE writes its external weights.
        epair_path.unlink()
        try:
            with external_call(
                "OfflineCompiler.compile",
                method=specs.method_name,
                acl_soc_name=specs.acl_soc_name,
            ):
                offline_compiler.compile(
                    graph,
                    output_base,
                    soc=specs.acl_soc_name,
                    options=build_options,
                )
        finally:
            del graph
        log_memory("AFTER_OFFLINE_COMPILE")

        om_path = _find_offline_model(output_base)
        log_memory("BEFORE_BUILD_MANIFEST")
        manifest = build_manifest(
            edge_program=edge_program,
            specs=specs,
            store=store,
            om_path=om_path,
            weight_dir=weight_dir,
        )
        log_memory("AFTER_BUILD_MANIFEST")
        # NamedDataStore owns the OM/weight bytes now. Do not retain a second
        # disk copy throughout compilation of the remaining model methods.
        shutil.rmtree(weight_dir)
        om_path.unlink()
        return manifest


def _find_offline_model(output_base: Path) -> Path:
    """Locate the compiled OM; GE may append the host os/arch to its name."""
    expected = output_base.with_suffix(".om")
    if expected.is_file():
        return expected
    candidates = sorted(output_base.parent.glob(f"{output_base.name}*.om"))
    if len(candidates) != 1:
        raise RuntimeError(f"Offline_compile did not produce expected OM: {expected}")
    return candidates[0]


def new_staging_root() -> tempfile.TemporaryDirectory[str]:
    """Create an auto-cleaned, process/thread-specific compilation workspace.

    Independent lowering calls can run concurrently in one process; including
    both identifiers prevents their EPAIR, OM, and weight paths from colliding.
    """
    prefix = f"executorch_npu_pid_{os.getpid()}_tid_{threading.get_native_id()}_"
    return tempfile.TemporaryDirectory(
        prefix=prefix, dir=os.environ.get("EXECUTORCH_NPU_STAGING", "/tmp")
    )


def _path_part(value: str) -> str:
    """Sanitize method names before using them in staging-directory paths."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
