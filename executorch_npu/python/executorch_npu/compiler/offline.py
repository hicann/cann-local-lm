# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Own the process-wide GE session used by loading and offline compilation."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from ..shared.compilation import compile_lock


class OfflineCompiler:
    """Lazily initialize one GE target and serialize its native operations."""

    def __init__(self) -> None:
        self._ge: ModuleType | None = None
        self._soc: str | None = None
        self._host_only: bool | None = None
        self._workspace: Path | None = None
        self._closed = False
        self._resources = ExitStack()

    def initialize(self, soc: str) -> None:
        """Initialize once for both EPAIR loading and model compilation."""
        with compile_lock:
            self._initialize(soc)

    def compile(
        self,
        graph: Any,
        output: Path,
        *,
        soc: str,
        options: Mapping[str, str],
    ) -> None:
        """Write the OM and external weights without changing the process cwd."""
        with compile_lock:
            output = output.absolute()
            self._initialize(soc)
            build_options = dict(options)
            # CANN accepts externalWeightDir only at session initialization.
            build_options.pop("ge.externalWeightDir", None)
            build_options.update(
                {"ge.externalWeight": "1", "ge.exec.precision_mode_v2": "origin"}
            )
            try:
                model = self._ge.build_model(graph, build_options)
                self._ge.save_model(str(output), model)
                self._move_weights(output.parent / "weight")
            finally:
                self._clear_weights()

    def _move_weights(self, destination: Path) -> None:
        """Transfer this build's files out of the session's external-weight area."""
        destination.mkdir(parents=True, exist_ok=True)
        for source in self._workspace.iterdir():
            # CANN's global weight directory also contains host-side bookkeeping.
            # meta.json is not a FileConstant payload and must not enter PTD.
            if source.name == "meta.json":
                continue
            target = destination / source.name
            if target.exists():
                raise FileExistsError(
                    f"External weight destination already exists: {target}"
                )
            # shutil.move also supports callers whose output is on another filesystem.
            shutil.move(str(source), str(target))

    def _clear_weights(self) -> None:
        """A failed build must not leave weights for the next partition."""
        for path in self._workspace.iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()

    def _initialize(self, soc: str) -> None:
        if self._closed:
            raise RuntimeError(
                "Offline compiler is closed; use a new process to compile again"
            )
        host_only = os.environ.get("EXECUTORCH_NPU_HOST_ONLY", "0") == "1"
        if self._ge is not None:
            if soc != self._soc:
                raise ValueError(
                    f"GE is initialized for {self._soc}; cannot switch to {soc}"
                )
            if host_only != self._host_only:
                raise ValueError(
                    "Cannot change EXECUTORCH_NPU_HOST_ONLY after GE initialization"
                )
            return

        import ge.offline_compile as ge

        try:
            with ExitStack() as resources:
                workspace = Path(
                    resources.enter_context(
                        tempfile.TemporaryDirectory(
                            prefix=f"executorch_npu_ge_{os.getpid()}_",
                            dir=os.environ.get("EXECUTORCH_NPU_STAGING", "/tmp"),
                        )
                    )
                )
                ge.build_initialize(
                    {
                        "ge.socVersion": soc,
                        "ge.graphRunMode": "0",
                        "ge.externalWeightDir": str(workspace),
                    }
                )
                resources.callback(ge.build_finalize)
                self._resources = resources.pop_all()
        except Exception:
            self._closed = True
            raise
        self._ge = ge
        self._soc = soc
        self._host_only = host_only
        self._workspace = workspace
        atexit.register(self.close)

    def close(self) -> None:
        """Finalize once before releasing the session's external-weight directory."""
        with compile_lock:
            self._closed = True
            self._ge = None
            atexit.unregister(self.close)
            self._resources.close()


offline_compiler = OfflineCompiler()
