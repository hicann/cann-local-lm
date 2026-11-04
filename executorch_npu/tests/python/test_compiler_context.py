# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import pytest

from executorch_npu.compiler import compatibility, delegate_adapter


@pytest.mark.parametrize("previous", [None, "1"])
def test_nested_fallback_scope_restores_on_exception(monkeypatch, previous):
    name = "TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN"
    if previous is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, previous)
    with delegate_adapter._fallback_codegen_disabled():
        with pytest.raises(RuntimeError, match="loader failed"):
            with delegate_adapter._fallback_codegen_disabled():
                assert os.environ[name] == "0"
                raise RuntimeError("loader failed")
        assert os.environ[name] == "0"
    assert os.environ.get(name) == previous


@pytest.mark.parametrize("fail_first", [False, True])
def test_overlapping_fallback_scopes_are_serialized_and_restore_environment(
    monkeypatch,
    fail_first,
):
    name = "TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN"
    monkeypatch.setenv(name, "1")
    entered, attempted, second_entered, release = [threading.Event() for _ in range(4)]

    def first():
        with delegate_adapter._fallback_codegen_disabled():
            entered.set()
            assert release.wait(5)
            assert os.environ[name] == "0"
            if fail_first:
                raise RuntimeError("loader failed")

    def second():
        attempted.set()
        with delegate_adapter._fallback_codegen_disabled():
            second_entered.set()
            assert os.environ[name] == "0"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first)
        try:
            assert entered.wait(5)
            second_result = pool.submit(second)
            assert attempted.wait(5)
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        if fail_first:
            with pytest.raises(RuntimeError, match="loader failed"):
                first_result.result(timeout=5)
        else:
            first_result.result(timeout=5)
        second_result.result(timeout=5)
    assert os.environ[name] == "1"


def test_compatibility_rejects_mode_change_before_mutating_backends(monkeypatch):
    calls = []
    monkeypatch.setattr(compatibility, "_configured_host_only", None)
    monkeypatch.setenv("EXECUTORCH_NPU_HOST_ONLY", "1")
    for name in (
        "_patch_torchair_deterministic_options",
        "_patch_torchair_host_only_graph_engine",
        "_patch_dynamo_compilation_metrics",
        "_patch_torch_delegate_scalar_serialization",
    ):
        monkeypatch.setattr(compatibility, name, lambda *args: calls.append("patch"))
    module = ModuleType("torch_delegate")
    monkeypatch.setitem(sys.modules, "torch_delegate", module)
    assert compatibility.load_torch_delegate() is module
    assert len(calls) == 4
    monkeypatch.setenv("EXECUTORCH_NPU_HOST_ONLY", "0")
    with pytest.raises(ValueError, match="EXECUTORCH_NPU_HOST_ONLY"):
        compatibility.load_torch_delegate()
    assert len(calls) == 4
