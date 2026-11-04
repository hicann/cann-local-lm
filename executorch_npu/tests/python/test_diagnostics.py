# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from types import SimpleNamespace

from executorch_npu.shared import diagnostics


def test_memory_diagnostics_do_no_work_when_disabled(monkeypatch, capsys):
    monkeypatch.delenv("ASCEND_EDGE_MEMORY_DIAGNOSTICS", raising=False)
    monkeypatch.setenv("ASCEND_EDGE_MEMORY_COLLECT_GC", "1")

    def unexpected(*args):
        raise AssertionError("disabled diagnostics touched the process")

    monkeypatch.setattr(diagnostics.psutil, "Process", unexpected)
    monkeypatch.setattr(diagnostics.gc, "collect", unexpected)
    diagnostics.log_memory("test")
    assert capsys.readouterr().out == ""


def test_memory_diagnostics_collect_only_when_explicitly_requested(monkeypatch, capsys):
    monkeypatch.setenv("ASCEND_EDGE_MEMORY_DIAGNOSTICS", "1")
    monkeypatch.delenv("ASCEND_EDGE_MEMORY_COLLECT_GC", raising=False)
    collected = []
    monkeypatch.setattr(diagnostics.gc, "collect", lambda: collected.append(True))
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1024, vms=2048))
    monkeypatch.setattr(diagnostics.psutil, "Process", lambda _: process)
    diagnostics.log_memory("sample")
    assert collected == []
    assert "[MEM:sample]" in capsys.readouterr().out
    monkeypatch.setenv("ASCEND_EDGE_MEMORY_COLLECT_GC", "1")
    diagnostics.log_memory("collect")
    assert collected == [True]


def test_partial_diagnostic_patch_setup_restores_previous_functions(monkeypatch):
    import executorch.exir.program._program as program_module
    import pytest

    monkeypatch.setenv("ASCEND_EDGE_MEMORY_DIAGNOSTICS", "1")
    original = program_module._gen_edge_manager_for_partitioners
    wrap = diagnostics._wrap
    count = 0

    def failing_wrap(module, name):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("incompatible instrumentation")
        return wrap(module, name)

    monkeypatch.setattr(diagnostics, "_wrap", failing_wrap)
    with pytest.raises(RuntimeError, match="incompatible instrumentation"):
        with diagnostics.edge_lowering_diagnostics():
            raise AssertionError("setup should have failed")
    assert program_module._gen_edge_manager_for_partitioners is original
