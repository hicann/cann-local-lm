# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Public ExecuTorch-NPU API with lightweight C++ SDK discovery."""

from __future__ import annotations

from typing import Any

from .shared.sdk import sdk_root


__all__ = [
    "AscendBackend",
    "Manifest",
    "AscendCompileOptions",
    "AscendPartitioner",
    "sdk_root",
    "aten_edge_compile_config",
    "build_state_io_contract",
]


def __getattr__(name: str) -> Any:
    """Load Python export APIs only when requested.

    C++ SDK consumers can locate ``sdk_root`` without installing Torch. Export
    APIs retain their existing top-level imports and load Torch/ExecuTorch on
    first access.
    """
    if name == "AscendBackend":
        from .backend import AscendBackend

        value = AscendBackend
    elif name == "Manifest":
        from .artifacts.manifest import Manifest

        value = Manifest
    elif name == "AscendCompileOptions":
        from .compiler.compile_spec import AscendCompileOptions

        value = AscendCompileOptions
    elif name == "AscendPartitioner":
        # ExecuTorch discovers Python backends from loaded BackendDetails
        # subclasses. Loading the partitioner must therefore also load the
        # matching backend before lowering starts.
        from .backend import AscendBackend
        from .partitioner import AscendPartitioner

        globals()["AscendBackend"] = AscendBackend
        value = AscendPartitioner
    elif name == "aten_edge_compile_config":
        from .edge_compile_config import aten_edge_compile_config

        value = aten_edge_compile_config
    elif name == "build_state_io_contract":
        from .artifacts.state import build_state_io_contract

        value = build_state_io_contract
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
