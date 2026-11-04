# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compatibility adapters for the validated TorchAir and torch-delegate stack."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any

import torch

from ..shared.compilation import compile_lock


def _patch_torchair_deterministic_options() -> None:
    """Keep standalone TorchAir's GE options valid for current CANN releases."""
    import torchair.core._backend as torchair_backend

    if getattr(
        torchair_backend,
        "_executorch_npu_deterministic_compat",
        False,
    ):
        return

    host_only = os.environ.get("EXECUTORCH_NPU_HOST_ONLY", "0") == "1"
    if host_only:
        # Host-side EPAIR conversion must not query torch_npu.npu.current_device().
        # GE still receives the target SoC through AscendCompileOptions.
        torchair_backend._get_device_id = lambda: int(
            os.environ.get("ASCEND_DEVICE_ID", "0")
        )

    original = torchair_backend._get_global_op_compile_config

    def get_compatible_options() -> dict[str, str]:
        if host_only:
            return {
                "ge.exec.allow_hf32": "00",
                "ge.deterministic": "0",
                "ge.deterministicLevel": "0",
            }
        options = original()
        if "torch_npu" not in sys.modules:
            options["ge.deterministic"] = "0"
            options["ge.deterministicLevel"] = "0"
        return options

    torchair_backend._get_global_op_compile_config = get_compatible_options
    torchair_backend._executorch_npu_deterministic_compat = True


def _patch_torch_delegate_scalar_serialization(torch_delegate: Any) -> None:
    """Handle scalar tensors in delegate's PT2 archive serializer."""
    writer = torch_delegate.epair.writer
    if getattr(writer, "_executorch_npu_scalar_compat", False):
        return
    original = writer._tensor_bytes

    def tensor_bytes(tensor: torch.Tensor) -> bytes:
        if tensor.dim() != 0:
            return original(tensor)
        return (
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )

    writer._tensor_bytes = tensor_bytes
    writer._executorch_npu_scalar_compat = True


def _patch_torchair_host_only_graph_engine() -> None:
    """Build GE IR without a device by stubbing TorchAir's native engine init."""
    if os.environ.get("EXECUTORCH_NPU_HOST_ONLY", "0") != "1":
        return

    import torchair._ge_concrete_graph.fx2ge_converter as fx2ge_converter
    import torchair.core._backend as torchair_backend

    if getattr(torchair_backend, "_executorch_npu_host_only_engine", False):
        return

    def _skip_native_engine(*args: Any, **kwargs: Any) -> None:
        return None

    torchair_backend._torchair.InitializeGraphEngine = _skip_native_engine
    torchair_backend._torchair.FinalizeGraphEngine = _skip_native_engine
    torchair_backend.initialize_graph_engine = _skip_native_engine
    torchair_backend.finalize_graph_engine = _skip_native_engine
    fx2ge_converter.initialize_graph_engine = _skip_native_engine
    torchair_backend._executorch_npu_host_only_engine = True


def _patch_dynamo_compilation_metrics() -> None:
    """Provide missing timer buckets required by older TorchAir integrations."""
    from torch._dynamo import utils as dynamo_utils

    metrics = dynamo_utils.compilation_time_metrics
    if not isinstance(metrics, defaultdict) or metrics.default_factory is not list:
        dynamo_utils.compilation_time_metrics = defaultdict(list, metrics)


_configured_host_only: bool | None = None


def load_torch_delegate() -> Any:
    """Install version adapters once under the compiler's global-state lock."""
    global _configured_host_only
    with compile_lock:
        host_only = os.environ.get("EXECUTORCH_NPU_HOST_ONLY", "0") == "1"
        if _configured_host_only is not None and host_only != _configured_host_only:
            raise ValueError(
                "Cannot change EXECUTORCH_NPU_HOST_ONLY after compiler compatibility setup"
            )
        _patch_torchair_deterministic_options()
        _configured_host_only = host_only
        _patch_torchair_host_only_graph_engine()
        _patch_dynamo_compilation_metrics()
        import torch_delegate

        _patch_torch_delegate_scalar_serialization(torch_delegate)
        return torch_delegate
