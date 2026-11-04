# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Opt-in Edge/state diagnostics and compilation phase reporting."""

from __future__ import annotations

import functools
import gc
import os
import time
from contextlib import contextmanager, ExitStack
from typing import Any, Callable, Iterator, Mapping, Sequence

import psutil

from .compilation import compile_lock


def _enabled() -> bool:
    """判断是否通过环境变量开启 Edge 降级阶段的内存诊断。"""
    return os.environ.get("ASCEND_EDGE_MEMORY_DIAGNOSTICS") == "1"


def _shape(result: Any) -> str:
    """提取多方法 Edge 程序的节点数量，生成便于日志检索的摘要。"""
    programs = getattr(result, "_edge_programs", None)
    if programs is None:
        programs = getattr(result, "_execution_programs", None)
    if not isinstance(programs, dict):
        return ""
    details = []
    for name, program in programs.items():
        graph = getattr(getattr(program, "graph_module", None), "graph", None)
        node_count = "?" if graph is None else sum(1 for _ in graph.nodes)
        details.append(f"{name}:nodes={node_count}")
    return " methods=" + ",".join(details)


def _log(tag: str, started: float, result: Any = None) -> None:
    """输出当前进程内存、耗时和可选的图规模信息。"""
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    elapsed = time.monotonic() - started
    print(
        f"[EDGE_MEM:{tag}] RSS={rss_mb:.1f}MB elapsed={elapsed:.3f}s{_shape(result)}",
        flush=True,
    )


def _wrap(module: Any, name: str) -> tuple[Any, str, Callable[..., Any]]:
    """临时包装目标函数，在调用前后记录 Edge 降级诊断信息。"""
    original = getattr(module, name)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """执行原函数并记录本次调用的内存变化。"""
        started = time.monotonic()
        _log(f"{name}:before", started)
        result = original(*args, **kwargs)
        _log(f"{name}:after", started, result)
        return result

    setattr(module, name, wrapped)
    return module, name, original


@contextmanager
def edge_lowering_diagnostics() -> Iterator[None]:
    """临时注入 ExecuTorch 降级边界诊断，退出上下文后恢复原函数。"""
    if not _enabled():
        yield
        return

    with compile_lock, ExitStack() as patches:
        import executorch.exir.backend.backend_api as backend_api
        import executorch.exir.program._program as program_module

        targets = [
            (program_module, "_gen_edge_manager_for_partitioners"),
            (program_module.EdgeProgramManager, "to_backend"),
            (program_module.EdgeProgramManager, "to_executorch"),
            (backend_api, "lower_all_submodules_to_backend"),
        ]
        for module, name in targets:
            module, name, original = _wrap(module, name)
            patches.callback(setattr, module, name, original)
        started = time.monotonic()
        _log("to_edge_transform_and_lower:before", started)
        try:
            yield
        finally:
            _log("to_edge_transform_and_lower:after", started)


def log_memory(tag: str) -> None:
    """记录编译流水线阶段的 RSS、VMS 和垃圾回收计数。"""
    if not _enabled():
        return
    if os.environ.get("ASCEND_EDGE_MEMORY_COLLECT_GC") == "1":
        gc.collect()
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / 1024 / 1024
    vms_mb = mem_info.vms / 1024 / 1024
    gc_counts = gc.get_count()
    print(
        f"[MEM:{tag}] RSS: {rss_mb:.1f}MB, VMS: {vms_mb:.1f}MB, GC counts: {gc_counts}",
        flush=True,
    )


def diagnostic(event: str, **fields: object) -> None:
    if os.environ.get("ASCEND_EXPORT_DIAGNOSTICS") != "1":
        return
    details = " ".join(f"{name}={value}" for name, value in fields.items())
    print(
        f"diagnostic={event} monotonic_s={time.monotonic():.3f} {details}", flush=True
    )


def phase_log(event: str, call: str, started: float, **fields: object) -> None:
    """输出外部编译调用的开始、结束或失败事件。"""
    details = " ".join(f"{name}={value}" for name, value in fields.items())
    elapsed = time.monotonic() - started
    suffix = f" {details}" if details else ""
    print(
        f"[COMPILE_PHASE] event={event} call={call} elapsed_s={elapsed:.3f}{suffix}",
        flush=True,
    )


@contextmanager
def external_call(name: str, **fields: object) -> Iterator[None]:
    started = time.monotonic()
    phase_log("start", name, started, **fields)
    diagnostic("external_start", call=name, **fields)
    try:
        yield
    except Exception as error:
        phase_log("failure", name, started, error=repr(error), **fields)
        diagnostic(
            "external_failure",
            call=name,
            elapsed_s=f"{time.monotonic() - started:.3f}",
            error=repr(error),
        )
        raise
    else:
        phase_log("end", name, started, **fields)
        diagnostic(
            "external_end",
            call=name,
            elapsed_s=f"{time.monotonic() - started:.3f}",
        )


@contextmanager
def ge_phase(name: str, **fields: object) -> Iterator[None]:
    """Report timing, memory usage and failures for one GE operation."""
    started = time.monotonic()
    _ge_phase_log("start", name, started, **fields)
    try:
        yield
    except Exception as error:
        _ge_phase_log("failure", name, started, error=repr(error), **fields)
        raise
    else:
        _ge_phase_log("end", name, started, **fields)


def _ge_phase_log(event: str, name: str, started: float, **fields: object) -> None:
    """Write the memory and elapsed time of a GE operation."""
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(
        f"[GE_PHASE] event={event} phase={name} rss_mb={rss_mb:.1f} "
        f"elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def log_state_mapping(
    input_names: Sequence[str],
    output_names: Sequence[str],
    nodes: Mapping[str, Any],
    dependencies: Mapping[str, set[str]],
) -> None:
    """Report state-mapping inputs without owning graph or contract rules."""
    if os.environ.get("ASCEND_DIAGNOSTIC_STATE_MAPPING") != "1":
        return
    print("\n=== Delegated Subgraph Structure ===", flush=True)
    print(f"Inputs ({len(input_names)}):", flush=True)
    for index, name in enumerate(input_names):
        tensor = nodes[name].meta.get("val") if name in nodes else None
        print(f"  [{index}] {name} shape={getattr(tensor, 'shape', None)}", flush=True)
    print(f"Outputs ({len(output_names)}):", flush=True)
    for index, name in enumerate(output_names):
        tensor = nodes[name].meta.get("val") if name in nodes else None
        print(
            f"  [{index}] {name} shape={getattr(tensor, 'shape', None)} "
            f"deps={sorted(dependencies[name])}",
            flush=True,
        )
    print("=== End Diagnostic ===\n", flush=True)
