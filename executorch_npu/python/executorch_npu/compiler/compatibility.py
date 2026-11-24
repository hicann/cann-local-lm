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


_configured_target_soc: str | None = None


def _patch_torchair_host_only_device_name(acl_soc_name: str) -> None:
    """Report the compile target instead of querying a physical device name.

    TorchAir converters such as ``native_batch_norm_legit_no_training`` and
    ``max_pool2d_with_indices`` call ``is_arch35()``, which reads the device
    name through ``torch_npu.npu.get_device_name()``. Without a device that
    query resolves no device index and raises ``TypeError``, which TorchAir
    does not catch. Supplying the target SoC keeps TorchAir's own architecture
    rule authoritative while removing the hardware dependency.
    """
    global _configured_target_soc
    if os.environ.get("EXECUTORCH_NPU_HOST_ONLY", "0") != "1":
        return
    if _configured_target_soc is not None and _configured_target_soc != acl_soc_name:
        raise ValueError(
            "Cannot change acl_soc_name after compiler compatibility setup"
        )
    _configured_target_soc = acl_soc_name
    npu = getattr(sys.modules.get("torch_npu"), "npu", None)
    if not hasattr(npu, "get_device_name"):
        # Without torch_npu, TorchAir's is_arch35() already reports False.
        return
    npu.get_device_name = lambda *args, **kwargs: acl_soc_name


def _patch_dynamo_compilation_metrics() -> None:
    """Provide missing timer buckets required by older TorchAir integrations."""
    from torch._dynamo import utils as dynamo_utils

    metrics = dynamo_utils.compilation_time_metrics
    if not isinstance(metrics, defaultdict) or metrics.default_factory is not list:
        dynamo_utils.compilation_time_metrics = defaultdict(list, metrics)


_configured_host_only: bool | None = None


def load_torch_delegate(acl_soc_name: str) -> Any:
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
        _patch_torchair_host_only_device_name(acl_soc_name)
        _patch_dynamo_compilation_metrics()
        import torch_delegate

        return torch_delegate
