# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest

from executorch_npu.compiler.offline import OfflineCompiler


class _FakeGe(ModuleType):
    def __init__(self) -> None:
        super().__init__("ge.offline_compile")
        self.calls: list[str] = []
        self.global_options: dict[str, str] = {}
        self.options: dict[str, str] = {}
        self.work_path: str | None = None
        self.cwd: str | None = None

    def build_initialize(self, options: dict[str, str]) -> None:
        self.calls.append("initialize")
        self.global_options = dict(options)

    def build_model(self, graph: object, options: dict[str, str]) -> object:
        self.calls.append("build")
        self.options = dict(options)
        self.work_path = os.environ.get("ASCEND_WORK_PATH")
        self.cwd = os.getcwd()
        (Path(self.global_options["ge.externalWeightDir"]) / "weight_0").write_bytes(
            b"weight"
        )
        return object()

    def save_model(self, output: str, model: object) -> None:
        self.calls.append("save")
        Path(output).with_suffix(".om").write_bytes(b"model")

    def build_finalize(self) -> None:
        self.calls.append("close")
        self.workspace_exists_at_close = Path(
            self.global_options["ge.externalWeightDir"]
        ).is_dir()


@pytest.fixture
def ge(monkeypatch):
    module = _FakeGe()
    package = ModuleType("ge")
    package.offline_compile = module
    monkeypatch.setitem(sys.modules, "ge", package)
    monkeypatch.setitem(sys.modules, "ge.offline_compile", module)
    return module


@pytest.fixture
def compiler(ge):
    compiler = OfflineCompiler()
    yield compiler
    compiler.close()


@pytest.fixture
def output(tmp_path):
    (tmp_path / "weight").mkdir()
    return tmp_path / "model"


@pytest.mark.parametrize("shape", ["256~512,4", "ge-owned-syntax"])
def test_build_options_are_forwarded_without_mutating_the_caller(
    compiler, ge, output, shape
):
    options = {
        "input_shape": shape,
        "ge.externalWeight": "0",
        "ge.externalWeightDir": "/unused",
    }
    original = dict(options)

    compiler.compile(object(), output, soc="Ascend910_9362", options=options)

    assert options == original
    assert ge.options == {
        "input_shape": shape,
        "ge.externalWeight": "1",
        "ge.exec.precision_mode_v2": "origin",
    }
    assert output.with_suffix(".om").read_bytes() == b"model"
    assert (output.parent / "weight" / "weight_0").read_bytes() == b"weight"


def test_ge_is_reused_and_finalized_once(compiler, ge, output):
    assert ge.calls == []
    for index in range(2):
        partition_output = output.parent / str(index) / "model"
        partition_output.parent.mkdir()
        compiler.compile(object(), partition_output, soc="Ascend910_9362", options={})
    assert Path(ge.global_options["ge.externalWeightDir"]).is_dir()

    compiler.close()
    compiler.close()

    assert ge.calls == ["initialize", "build", "save", "build", "save", "close"]
    assert ge.workspace_exists_at_close
    assert not Path(ge.global_options["ge.externalWeightDir"]).exists()
    with pytest.raises(RuntimeError, match="closed"):
        compiler.compile(object(), output, soc="Ascend910_9362", options={})


def test_close_before_first_use_does_not_initialize_ge(compiler, ge, output):
    compiler.close()
    with pytest.raises(RuntimeError, match="closed"):
        compiler.compile(object(), output, soc="Ascend910_9362", options={})
    assert ge.calls == []


def test_initialized_session_rejects_a_different_soc(compiler, ge, output):
    compiler.compile(object(), output, soc="Ascend910_9362", options={})
    with pytest.raises(ValueError, match="cannot switch"):
        compiler.compile(object(), output, soc="Ascend310P3", options={})
    assert ge.calls == ["initialize", "build", "save"]


def test_initialization_failure_prevents_retry(compiler, ge, output, monkeypatch):
    initialize = ge.build_initialize

    def fail(options):
        initialize(options)
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(ge, "build_initialize", fail)
    with pytest.raises(RuntimeError, match="initialization failed"):
        compiler.compile(object(), output, soc="Ascend910_9362", options={})
    with pytest.raises(RuntimeError, match="closed"):
        compiler.compile(object(), output, soc="Ascend910_9362", options={})
    compiler.close()
    assert ge.calls == ["initialize"]
    assert not Path(ge.global_options["ge.externalWeightDir"]).exists()


def test_close_failure_does_not_repeat_native_cleanup(
    compiler, ge, output, monkeypatch
):
    finalize = ge.build_finalize

    def fail():
        finalize()
        raise RuntimeError("close failed")

    monkeypatch.setattr(ge, "build_finalize", fail)
    compiler.compile(object(), output, soc="Ascend910_9362", options={})
    with pytest.raises(RuntimeError, match="close failed"):
        compiler.close()
    compiler.close()
    assert ge.calls.count("close") == 1
    assert ge.workspace_exists_at_close
    assert not Path(ge.global_options["ge.externalWeightDir"]).exists()
    with pytest.raises(RuntimeError, match="closed"):
        compiler.compile(object(), output, soc="Ascend910_9362", options={})


@pytest.mark.parametrize("operation", ["build_model", "save_model"])
def test_failed_compilation_restores_cwd_and_work_path(
    compiler, ge, output, monkeypatch, operation
):
    cwd = os.getcwd()
    monkeypatch.setenv("ASCEND_WORK_PATH", "/tmp/existing-knowledge-base")

    def fail(*args):
        raise RuntimeError("compilation failed")

    monkeypatch.setattr(ge, operation, fail)
    with pytest.raises(RuntimeError, match="compilation failed"):
        compiler.compile(object(), output, soc="Ascend910_9362", options={})
    assert os.getcwd() == cwd
    assert os.environ["ASCEND_WORK_PATH"] == "/tmp/existing-knowledge-base"


def test_relative_output_and_custom_staging_preserve_work_path(
    compiler, ge, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    (tmp_path / "weight").mkdir()
    monkeypatch.setenv("EXECUTORCH_NPU_STAGING", str(staging))
    monkeypatch.setenv("ASCEND_WORK_PATH", "/tmp/existing-knowledge-base")

    compiler.compile(object(), Path("model"), soc="Ascend910_9362", options={})

    assert (tmp_path / "model.om").is_file()
    assert Path(ge.cwd) == tmp_path
    assert "ge.externalWeightDir" not in ge.options
    assert Path(ge.global_options["ge.externalWeightDir"]).parent == staging
    assert Path(ge.global_options["ge.externalWeightDir"]).is_dir()
    assert ge.work_path == "/tmp/existing-knowledge-base"
    assert Path.cwd() == tmp_path


def test_process_exit_finalizes_ge(tmp_path):
    script = r"""
import pathlib
import sys
import types
from executorch_npu.compiler.offline import offline_compiler

ge = types.ModuleType("ge")
offline = types.ModuleType("ge.offline_compile")
ge.offline_compile = offline
sys.modules["ge"] = ge
sys.modules["ge.offline_compile"] = offline
offline.build_initialize = lambda options: None
offline.build_finalize = lambda: pathlib.Path(sys.argv[1]).write_text("finalized")
offline_compiler.initialize("test")
"""
    record = tmp_path / "finalize.txt"
    subprocess.run(
        [sys.executable, "-c", script, str(record)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert record.read_text() == "finalized"


def test_staging_root_uses_process_thread_and_random_suffix(monkeypatch):
    monkeypatch.delenv("EXECUTORCH_NPU_STAGING", raising=False)
    from executorch_npu.compiler import new_staging_root

    prefix = f"executorch_npu_pid_{os.getpid()}_tid_{threading.get_native_id()}_"
    with new_staging_root() as root:
        path = Path(root)
        assert path.parent == Path("/tmp")
        assert path.name.startswith(prefix)
        assert len(path.name) > len(prefix)
    assert not path.exists()


def test_epair_initialization_and_offline_build_share_one_session(
    compiler, ge, output, monkeypatch
):
    from executorch_npu.compiler import delegate_adapter

    monkeypatch.setenv("EXECUTORCH_NPU_HOST_ONLY", "1")
    monkeypatch.setattr(delegate_adapter, "offline_compiler", compiler)
    delegate_adapter._initialize_ge("Ascend910_9362")
    compiler.compile(object(), output, soc="Ascend910_9362", options={})
    assert ge.calls == ["initialize", "build", "save"]
    compiler.close()
    assert ge.calls == ["initialize", "build", "save", "close"]


def test_initialized_session_rejects_a_different_host_mode(compiler, ge, monkeypatch):
    monkeypatch.setenv("EXECUTORCH_NPU_HOST_ONLY", "1")
    compiler.initialize("Ascend910_9362")
    monkeypatch.setenv("EXECUTORCH_NPU_HOST_ONLY", "0")
    with pytest.raises(ValueError, match="EXECUTORCH_NPU_HOST_ONLY"):
        compiler.initialize("Ascend910_9362")
    assert ge.calls == ["initialize"]


def test_each_partition_owns_its_external_weights(compiler, ge, tmp_path):
    options = {"ge.externalWeightDir": "/ignored"}
    for name in ("prefill", "decode"):
        directory = tmp_path / name
        (directory / "weight").mkdir(parents=True)
        compiler.compile(
            object(), directory / "model", soc="Ascend910_9362", options=options
        )
        assert "ge.externalWeightDir" not in ge.options
        assert not list(Path(ge.global_options["ge.externalWeightDir"]).iterdir())
        assert (directory / "weight" / "weight_0").read_bytes() == b"weight"
    assert options == {"ge.externalWeightDir": "/ignored"}
    assert ge.calls.count("initialize") == 1


def test_build_keeps_cwd_visible_to_other_threads_and_close_waits(
    compiler,
    ge,
    output,
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor

    original_cwd = os.getcwd()
    started, release, closing = threading.Event(), threading.Event(), threading.Event()
    build = ge.build_model

    def blocked_build(*args):
        # Initialization can be reached again from a native adapter callback.
        compiler.initialize("Ascend910_9362")
        started.set()
        assert release.wait(5)
        return build(*args)

    def close():
        closing.set()
        compiler.close()

    monkeypatch.setattr(ge, "build_model", blocked_build)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            compiler.compile, object(), output, soc="Ascend910_9362", options={}
        )
        try:
            assert started.wait(5)
            assert os.getcwd() == original_cwd
            second = pool.submit(close)
            assert closing.wait(5)
            assert not second.done()
        finally:
            release.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert ge.calls == ["initialize", "build", "save", "close"]


def test_failed_build_does_not_leak_weights_into_the_next_partition(
    compiler,
    ge,
    output,
    monkeypatch,
):
    original = ge.build_model

    def fail(*args):
        original(*args)
        (Path(ge.global_options["ge.externalWeightDir"]) / "partial").write_bytes(
            b"bad"
        )
        raise RuntimeError("partial build")

    with monkeypatch.context() as patch:
        patch.setattr(ge, "build_model", fail)
        with pytest.raises(RuntimeError, match="partial build"):
            compiler.compile(object(), output, soc="Ascend910_9362", options={})
    assert not list(Path(ge.global_options["ge.externalWeightDir"]).iterdir())
    compiler.compile(object(), output, soc="Ascend910_9362", options={})
    assert sorted(path.name for path in (output.parent / "weight").iterdir()) == [
        "weight_0"
    ]


def test_weight_transfer_does_not_overwrite_existing_output(compiler, ge, output):
    (output.parent / "weight" / "weight_0").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="already exists"):
        compiler.compile(object(), output, soc="Ascend910_9362", options={})
    assert (output.parent / "weight" / "weight_0").read_bytes() == b"existing"
    assert not list(Path(ge.global_options["ge.externalWeightDir"]).iterdir())


def test_session_metadata_is_not_packaged_as_a_weight(
    compiler, ge, output, monkeypatch
):
    from executorch.exir._serialize._named_data_store import NamedDataStore

    from executorch_npu.artifacts.builder import collect_and_register_weights

    build_model = ge.build_model

    def build_with_metadata(*args):
        model = build_model(*args)
        workspace = Path(ge.global_options["ge.externalWeightDir"])
        (workspace / "meta.json").write_text('{"weight_file_path": "session metadata"}')
        # File extensions do not identify tensor payloads; only CANN's reserved
        # bookkeeping filename must be excluded.
        (workspace / "opaque_weight.json").write_bytes(b"tensor-bytes")
        return model

    monkeypatch.setattr(ge, "build_model", build_with_metadata)
    compiler.compile(object(), output, soc="Ascend910_9362", options={})
    artifacts = collect_and_register_weights(
        NamedDataStore(), output.parent / "weight", "weights"
    )
    assert {artifact.file_name for artifact in artifacts} == {
        "weight_0",
        "opaque_weight.json",
    }
    assert not list(Path(ge.global_options["ge.externalWeightDir"]).iterdir())
