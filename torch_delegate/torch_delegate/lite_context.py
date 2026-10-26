# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Process-scoped context for EPAIR export without CANN runtime libraries."""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import enum
import importlib.machinery
import importlib.util
import os
import sys
import threading
from typing import Any, Iterator

from .core.utils import logger
from .fake_module_builder import FakeModuleBuilder


class LiteExportError(RuntimeError):
    """Base class for EPAIR export context failures."""


class LiteConfigError(LiteExportError):
    """The requested export configuration is invalid."""


class LiteImportError(LiteExportError):
    """The process has already loaded a conflicting native dependency."""


class LiteStateError(LiteExportError):
    """The requested operation conflicts with process or context state."""


class LiteDependencyError(LiteExportError):
    """A dependency required by a converter is unavailable in Lite mode."""


class LiteConversionError(LiteExportError):
    """A TorchAir conversion failed while Lite mode was active."""

    @classmethod
    def from_exception(cls, exc: Exception) -> "LiteConversionError":
        return cls(
            "Lite EPAIR conversion failed during TorchAir graph construction: "
            f"{type(exc).__name__}: {exc}. Check the TorchAir/torch_npu "
            "combination or use a clean Full-mode process."
        )


@dataclasses.dataclass(frozen=True)
class LiteConfig:
    """Target information used while constructing EPAIR IR."""

    target_soc: str
    allow_hf32: bool = False
    target_cann_version: str | None = None


class ProcessMode(enum.Enum):
    CLEAN = "clean"
    LITE_INITIALIZED = "lite_initialized"
    FULL_INITIALIZED = "full_initialized"
    CONFLICTED = "conflicted"


_PROCESS_MODE = ProcessMode.CLEAN
_PROCESS_CONFIG: LiteConfig | None = None
_VENDORED_TORCHAIR_PATH: str | None = None
_ACTIVE_CONFIG: contextvars.ContextVar[LiteConfig | None] = contextvars.ContextVar(
    "torch_delegate_active_epair_config", default=None
)
_VIOLATIONS: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("torch_delegate_lite_violations", default=None)
)
_CONTEXT_DEPTH = 0
_OWNER_THREAD: int | None = None
_LITE_THREAD: int | None = None
_STATE_LOCK = threading.RLock()
_EXPORT_LOCK = threading.Lock()


def _loaded_conflicts() -> list[str]:
    conflicts = []
    for name in ("torchair", "torchair.core", "torchair.core._torchair", "torch_npu"):
        if name in sys.modules:
            conflicts.append(name)
    return conflicts


def _validate_config(config: LiteConfig) -> LiteConfig:
    if not isinstance(config, LiteConfig):
        raise LiteConfigError("lite_export_context requires LiteConfig")
    if not isinstance(config.target_soc, str) or not config.target_soc:
        raise LiteConfigError("target_soc must be a non-empty string")
    if not isinstance(config.allow_hf32, bool):
        raise LiteConfigError("allow_hf32 must be bool")
    if config.target_cann_version is not None and not isinstance(
        config.target_cann_version, str
    ):
        raise LiteConfigError("target_cann_version must be a string or None")
    return config


def _initialize_process(config: LiteConfig) -> None:
    global _PROCESS_MODE, _PROCESS_CONFIG, _VENDORED_TORCHAIR_PATH
    with _STATE_LOCK:
        if _PROCESS_MODE is ProcessMode.LITE_INITIALIZED:
            if config != _PROCESS_CONFIG:
                raise LiteStateError(
                    "Lite EPAIR export is already initialized with a different "
                    "configuration; use a separate process for another target"
                )
            return
        if _PROCESS_MODE is not ProcessMode.CLEAN:
            raise LiteStateError(
                f"cannot enter Lite EPAIR export from process mode "
                f"{_PROCESS_MODE.value}; start a clean process"
            )
        conflicts = _loaded_conflicts()
        if conflicts:
            _PROCESS_MODE = ProcessMode.CONFLICTED
            raise LiteImportError(
                "Lite EPAIR export must start before importing TorchAir or "
                f"torch_npu; conflicting modules: {', '.join(conflicts)}. "
                "Set TORCH_DEVICE_BACKEND_AUTOLOAD=0 before importing torch "
                "and restart the process."
            )
        _VENDORED_TORCHAIR_PATH = _find_vendored_torchair_path()
        _install_stubs(config)
        _PROCESS_CONFIG = config
        _PROCESS_MODE = ProcessMode.LITE_INITIALIZED


def _find_vendored_torchair_path() -> str | None:
    torch_npu_spec = importlib.util.find_spec("torch_npu")
    if torch_npu_spec is None or not torch_npu_spec.submodule_search_locations:
        return None
    torch_npu_path = next(iter(torch_npu_spec.submodule_search_locations))
    candidate = f"{torch_npu_path}/dynamo"
    torchair_spec = importlib.machinery.PathFinder.find_spec("torchair", [candidate])
    return candidate if torchair_spec is not None else None


def _install_stubs(config: LiteConfig) -> None:
    """Install the registered native and torch_npu fake module tree."""
    modules = FakeModuleBuilder(config, _unsupported).build()
    npu = modules["torch_npu.npu"]
    import torch

    torch.npu = npu
    logger.debug("Installed Lite TorchAir native stub for %s", config.target_soc)


def _unsupported(symbol: str, record_violation: bool = True):
    def reject(*args, **kwargs):
        violations = _VIOLATIONS.get()
        if record_violation and violations is not None:
            violations.append({"symbol": symbol, "args": args, "kwargs": kwargs})
        raise LiteDependencyError(
            f"Lite EPAIR export does not support native dependency call: {symbol}"
        )

    return reject


def mark_full_initialized() -> None:
    global _PROCESS_MODE
    with _STATE_LOCK:
        if _PROCESS_MODE is ProcessMode.LITE_INITIALIZED:
            raise LiteStateError(
                "the process was initialized for Lite EPAIR export; "
                "use a separate process for Full export"
            )
        if _PROCESS_MODE is ProcessMode.CLEAN:
            _PROCESS_MODE = ProcessMode.FULL_INITIALIZED


def active_config() -> LiteConfig | None:
    return _ACTIVE_CONFIG.get()


def check_violations() -> None:
    violations = _VIOLATIONS.get() or []
    if violations:
        first = violations[0]
        raise LiteDependencyError(
            "Lite EPAIR conversion used an unsupported dependency "
            f"{first['symbol']}; export this model in a clean Full-mode process"
        )


def validate_lite_result(ep_json: dict, total_nodes: int, converted_nodes: int) -> None:
    if converted_nodes != total_nodes:
        raise LiteConversionError(
            "Lite EPAIR conversion cannot contain unconverted nodes: "
            f"converted {converted_nodes}/{total_nodes}"
        )
    graph_nodes = ep_json.get("graph_module", {}).get("graph", {}).get("nodes", [])
    for node in graph_nodes:
        ascend_ir = node.get("ascend_ir")
        if not isinstance(ascend_ir, dict) or ascend_ir.get("source") != "converter":
            raise LiteConversionError(
                "Lite EPAIR conversion cannot contain fallback or unannotated node: "
                f"{node.get('target', '<unknown>')}"
            )


def prepare_lite_torchair() -> None:
    """Remove TorchAir's device-executor lookup from the build-only path."""
    config = _ACTIVE_CONFIG.get()
    if config is None:
        return
    try:
        _load_lite_torchair()
        from torchair.core import _backend
    except Exception as exc:
        raise LiteImportError(
            "failed to import TorchAir while preparing Lite EPAIR export; "
            "verify the TorchAir Python package and native stub coverage"
        ) from exc
    _backend._get_device_id = lambda: 0
    _backend._get_global_op_compile_config = lambda: {
        "ge.exec.allow_hf32": "1" if config.allow_hf32 else "0",
        "ge.deterministic": "1",
        "ge.deterministicLevel": "0",
    }


def _load_lite_torchair() -> None:
    if "torchair" in sys.modules:
        return
    if _VENDORED_TORCHAIR_PATH is None:
        return
    spec = importlib.machinery.PathFinder.find_spec(
        "torchair", [_VENDORED_TORCHAIR_PATH]
    )
    if spec is None or spec.loader is None:
        raise LiteImportError("failed to locate vendored TorchAir Python package")
    module = importlib.util.module_from_spec(spec)
    sys.modules["torchair"] = module
    core = sys.modules.get("torchair.core")
    if core is not None:
        core.__path__ = [os.path.join(_VENDORED_TORCHAIR_PATH, "torchair", "core")]
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("torchair", None)
        raise


def export_lock() -> threading.Lock:
    return _EXPORT_LOCK


@contextlib.contextmanager
def lite_export_context(config: LiteConfig) -> Iterator[None]:
    """Activate process-scoped, build-only EPAIR export mode."""
    global _CONTEXT_DEPTH, _OWNER_THREAD, _LITE_THREAD
    validated = _validate_config(config)
    current_thread = threading.get_ident()
    with _STATE_LOCK:
        if _LITE_THREAD is not None and _LITE_THREAD != current_thread:
            raise LiteStateError(
                "Lite EPAIR export does not support another thread in the same "
                "process; use a separate process"
            )
        if _OWNER_THREAD is not None and _OWNER_THREAD != current_thread:
            raise LiteStateError("Lite EPAIR export context does not support threads")
        _initialize_process(validated)
        if _CONTEXT_DEPTH and validated != _PROCESS_CONFIG:
            raise LiteStateError("nested Lite contexts must use the same configuration")
        _OWNER_THREAD = current_thread
        _LITE_THREAD = current_thread
        _CONTEXT_DEPTH += 1
    token = _ACTIVE_CONFIG.set(validated)
    violation_token = _VIOLATIONS.set([])
    try:
        yield
    finally:
        _VIOLATIONS.reset(violation_token)
        _ACTIVE_CONFIG.reset(token)
        with _STATE_LOCK:
            _CONTEXT_DEPTH -= 1
            if _CONTEXT_DEPTH == 0:
                _OWNER_THREAD = None


def process_mode() -> ProcessMode:
    return _PROCESS_MODE
