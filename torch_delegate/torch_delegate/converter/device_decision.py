# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Device 归属判定报告与严格校验。

归属报告：导出结束打印逐节点 Device 归属清单，按 namespace 分级告警。
严格校验：``require_aten_on_npu=True`` 时 aten 算子落到 CPU 直接报错。
两者都只做提示/拦截，不改变归属结论本身。
"""

from ..core.utils import logger
from ..epair.schema import DeviceDecision

_REPORT_SEPARATOR = "─"


def _render_group(title: str, entries: list[str]) -> str:
    lines = [f"  {title} ({len(entries)}):"]
    lines.extend(f"    {entry}" for entry in entries)
    return "\n".join(lines)


def format_device_report(decisions: list[DeviceDecision]) -> str:
    """渲染完整归属报告（NPU 与 CPU 两侧都列全，CPU 侧按 namespace 分级）。

    分级规则：
    - CPU（cpu_only）+ aten::      → WARNING（torch_npu 契约应覆盖 aten，缺失属异常）
    - CPU（cpu_only）+ 自定义 ns   → INFO（用户可能本就只写了 CPU 实现，属预期）
    - NPU（converter/has_privateuse1） → INFO（正常）
    """
    npu_converter: list[str] = []
    npu_fallback: list[str] = []
    cpu_suspicious: list[str] = []
    cpu_expected: list[str] = []
    for d in decisions:
        label = f"{d.op_name}.{d.overload or 'default'}  has_privateuse1={d.has_privateuse1}  has_cpu={d.has_cpu}"
        if d.rule == "converter":
            npu_converter.append(label)
        elif d.rule == "has_privateuse1":
            npu_fallback.append(label)
        elif d.namespace == "aten":
            cpu_suspicious.append(label)
        else:
            cpu_expected.append(label)

    parts = [
        f"[torch_delegate] Device ownership report: {len(decisions)} compute nodes"
    ]
    npu_total = len(npu_converter) + len(npu_fallback)
    parts.append("")
    parts.append(
        f"  {_REPORT_SEPARATOR * 4} NPU ({npu_total}) {_REPORT_SEPARATOR * 24}"
    )
    if npu_converter:
        parts.append(_render_group("converter -> GE built-in", npu_converter))
    if npu_fallback:
        parts.append(_render_group("fallback -> PrivateUse1", npu_fallback))
    parts.append("")
    parts.append(
        f"  {_REPORT_SEPARATOR * 4} CPU ({len(cpu_suspicious) + len(cpu_expected)}) {_REPORT_SEPARATOR * 24}"
    )
    if cpu_suspicious:
        parts.append(
            _render_group(
                "suspicious: aten op has no PrivateUse1 kernel", cpu_suspicious
            )
        )
    if cpu_expected:
        parts.append(
            _render_group("expected: custom namespace with CPU-only impl", cpu_expected)
        )
    return "\n".join(parts)


def log_device_report(decisions: list[DeviceDecision]) -> None:
    """打印归属报告；CPU 侧按 namespace 分级输出 WARNING/INFO。"""
    if not decisions:
        return
    for line in format_device_report(decisions).splitlines():
        logger.info(line)
    for d in decisions:
        if d.device == "cpu" and d.namespace == "aten":
            logger.warning(
                "aten op %s (%s) has no PrivateUse1 kernel and will run on CPU; "
                "this usually indicates a torch_npu coverage gap or version mismatch",
                d.op_name,
                d.overload or "default",
            )


def validate_device_decisions(
    decisions: list[DeviceDecision], require_aten_on_npu: bool
) -> None:
    """严格校验开关：aten 命名空间算子落到 CPU 时导出报错，一次性列出全部命中。"""
    if not require_aten_on_npu:
        return
    hits = [d for d in decisions if d.device == "cpu" and d.namespace == "aten"]
    if not hits:
        return
    detail = "\n".join(
        f"  {d.node_name}: {d.op_name}.{d.overload or 'default'} has_privateuse1={d.has_privateuse1} has_cpu={d.has_cpu}"
        for d in hits
    )
    raise RuntimeError(
        f"require_aten_on_npu=True: {len(hits)} aten op(s) resolved to CPU execution "
        "(no PrivateUse1 kernel). Export aborted. Ops:\n" + detail
    )
