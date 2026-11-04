# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

from pathlib import Path

from executorch.exir._serialize._named_data_store import NamedDataStore
from executorch.exir.backend.backend_details import (
    BackendDetails,
    PreprocessResult,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from torch.export import ExportedProgram

from .artifacts.builder import check_manifest_set
from .artifacts.manifest import Manifest
from .compiler import compile_partition, new_staging_root
from .compiler.compile_spec import AscendCompileOptions


class AscendBackend(BackendDetails):
    """ExecuTorch lowering entry points for the Ascend backend."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: list[CompileSpec],
    ) -> PreprocessResult:
        """Compile one delegated partition and return its referenced named data."""
        specs = AscendCompileOptions.model_validate_json(bytes(compile_specs[0].value))
        store = NamedDataStore()
        with new_staging_root() as staging:
            manifest = compile_partition(edge_program, specs, store, Path(staging))
        return PreprocessResult(
            processed_bytes=manifest.to_bytes(),
            data_store_output=store.get_named_data_store_output(),
        )

    @classmethod
    def preprocess_multimethod(
        cls,
        edge_programs: dict[str, list[ExportedProgram]],
        compile_specs: dict[str, list[list[CompileSpec]]],
    ) -> dict[str, list[PreprocessResult]]:
        """Compile all methods against one store and validate shared state.

        A single ``NamedDataStore`` lets content-addressed models and weights be
        emitted once even when prefill and decode partitions reference the same
        payload. Keep typed manifests until their shared-state metadata and reset
        ownership agree, then serialize each delegate exactly once.
        """
        store = NamedDataStore()
        manifests_by_method: dict[str, list[Manifest]] = {}
        with new_staging_root() as staging:
            staging_root = Path(staging)
            for method_name, programs in edge_programs.items():
                manifests_by_method[method_name] = []
                for program, raw_specs in zip(
                    programs, compile_specs[method_name], strict=True
                ):
                    specs = AscendCompileOptions.model_validate_json(
                        bytes(raw_specs[0].value)
                    )
                    manifest = compile_partition(program, specs, store, staging_root)
                    manifests_by_method[method_name].append(manifest)

        check_manifest_set(
            manifest for values in manifests_by_method.values() for manifest in values
        )
        output = store.get_named_data_store_output()
        return {
            method_name: [
                PreprocessResult(
                    processed_bytes=manifest.to_bytes(),
                    data_store_output=output,
                )
                for manifest in manifests
            ]
            for method_name, manifests in manifests_by_method.items()
        }
