# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""torch_delegate.EpairModelLoader Python 接口 UT。

通过 C ABI stub so（tests/python/stub/epair_model_loader_c_stub.c，经
TorchDelegateTestStubs.cmake 构建目标 td_epair_loader_c_stub 产出）注入
成功/各失败码，覆盖全部分支，无需真实 .epair 与设备。
未构建 stub so 时整组用例跳过。
"""

import ctypes
import glob
import os
import pathlib
import shutil
import sys
import threading
import types

import pytest

import torch_delegate
from torch_delegate import _c_abi
from torch_delegate import model_loader
from torch_delegate.model_loader import EpairModelLoader, EpairModelLoaderError

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

STUB_SEARCH_PATHS = (
    REPO_ROOT
    / "build_tests"
    / "tests"
    / "python"
    / "epair_loader_stub"
    / "lib64"
    / "libtd_epair_loader_c_stub.so",
    REPO_ROOT
    / "build"
    / "tests"
    / "python"
    / "epair_loader_stub"
    / "lib64"
    / "libtd_epair_loader_c_stub.so",
)

EPAIR_PATH = "/tmp/fake_model.epair"
STUB_GRAPH_PTR = 0x1000


def _find_stub_so() -> str | None:
    from_env = os.environ.get("TORCH_DELEGATE_EPAIR_LOADER_STUB_SO")
    if from_env:
        return from_env
    for path in STUB_SEARCH_PATHS:
        if path.is_file():
            return str(path)
    return None


STUB_SO = _find_stub_so()

pytestmark = pytest.mark.skipif(
    STUB_SO is None,
    reason="C ABI stub so 未构建，请先执行 ./build_test.sh 或设置 TORCH_DELEGATE_EPAIR_LOADER_STUB_SO",
)


def _load_stub_library() -> ctypes.CDLL:
    library = ctypes.CDLL(STUB_SO)
    _c_abi._declare_abi(library)
    return library


def _destroy_count(library: ctypes.CDLL) -> int:
    library.TdEpairStubDestroyCount.argtypes = []
    library.TdEpairStubDestroyCount.restype = ctypes.c_uint64
    return library.TdEpairStubDestroyCount()


def _fake_graph_factory(calls: list):
    def factory():
        graph = object()
        calls.append(graph)
        return graph, STUB_GRAPH_PTR

    return factory


@pytest.fixture
def stub_library(monkeypatch):
    library = _load_stub_library()
    monkeypatch.setattr(_c_abi, "_load_backend_library", lambda: library)
    return library


# ===========================================================================
# 功能：with 上下文加载与取图
# ===========================================================================


def test_with_context_returns_graph(stub_library, monkeypatch):
    calls = []
    monkeypatch.setattr(model_loader, "_new_ge_graph", _fake_graph_factory(calls))
    before = _destroy_count(stub_library)
    with EpairModelLoader(EPAIR_PATH) as loader:
        graph = loader.get_ge_graph()
        assert graph is calls[-1]
    assert _destroy_count(stub_library) == before + 1


def test_get_ge_graph_returns_independent_copies(stub_library, monkeypatch):
    calls = []
    monkeypatch.setattr(model_loader, "_new_ge_graph", _fake_graph_factory(calls))
    loader = EpairModelLoader(EPAIR_PATH)
    try:
        first = loader.get_ge_graph()
        second = loader.get_ge_graph()
        assert first is not second
        assert len(calls) == 2
    finally:
        loader.close()


# ===========================================================================
# 功能：生命周期管理
# ===========================================================================


def test_get_ge_graph_after_close_raises(stub_library, monkeypatch):
    monkeypatch.setattr(model_loader, "_new_ge_graph", _fake_graph_factory([]))
    loader = EpairModelLoader(EPAIR_PATH)
    loader.close()
    with pytest.raises(EpairModelLoaderError, match="closed"):
        loader.get_ge_graph()


def test_close_is_idempotent(stub_library):
    loader = EpairModelLoader(EPAIR_PATH)
    before = _destroy_count(stub_library)
    loader.close()
    loader.close()
    loader.close()
    assert _destroy_count(stub_library) == before + 1


def test_failed_construction_does_not_destroy(stub_library, monkeypatch):
    monkeypatch.setenv("TD_STUB_CREATE_STATUS", "1")
    before = _destroy_count(stub_library)
    with pytest.raises(EpairModelLoaderError):
        EpairModelLoader(EPAIR_PATH)
    # C ABI Create 失败时 handle 为 NULL，不应触发 Destroy
    assert _destroy_count(stub_library) == before


def test_exception_inside_with_still_closes(stub_library, monkeypatch):
    monkeypatch.setattr(model_loader, "_new_ge_graph", _fake_graph_factory([]))
    before = _destroy_count(stub_library)
    with pytest.raises(RuntimeError, match="boom"):
        with EpairModelLoader(EPAIR_PATH):
            raise RuntimeError("boom")
    assert _destroy_count(stub_library) == before + 1


def test_close_does_not_raise_when_destroy_fails(stub_library, monkeypatch):
    loader = EpairModelLoader(EPAIR_PATH)

    def broken_load_library():
        raise EpairModelLoaderError("backend library unavailable at destroy time")

    monkeypatch.setattr(_c_abi, "_load_backend_library", broken_load_library)
    loader.close()
    loader.close()


# ===========================================================================
# 错误上报：状态码到异常的映射
# ===========================================================================


def test_create_invalid_param_maps_to_value_error(stub_library, monkeypatch):
    monkeypatch.setenv("TD_STUB_CREATE_STATUS", "2")
    with pytest.raises(ValueError, match="INVALID_PARAM"):
        EpairModelLoader(EPAIR_PATH)


def test_create_failed_maps_to_loader_error_with_detail(stub_library, monkeypatch):
    monkeypatch.setenv("TD_STUB_CREATE_STATUS", "1")
    with pytest.raises(
        EpairModelLoaderError, match="stub error: Create failed: status=FAILED"
    ):
        EpairModelLoader(EPAIR_PATH)


def test_create_internal_error_maps_to_loader_error(stub_library, monkeypatch):
    monkeypatch.setenv("TD_STUB_CREATE_STATUS", "3")
    with pytest.raises(EpairModelLoaderError, match="INTERNAL_ERROR"):
        EpairModelLoader(EPAIR_PATH)


def test_get_graph_failure_maps_to_loader_error(stub_library, monkeypatch):
    monkeypatch.setattr(model_loader, "_new_ge_graph", _fake_graph_factory([]))
    with EpairModelLoader(EPAIR_PATH) as loader:
        monkeypatch.setenv("TD_STUB_GET_GRAPH_STATUS", "1")
        with pytest.raises(
            EpairModelLoaderError, match="stub error: GetGraph failed: status=FAILED"
        ):
            loader.get_ge_graph()


# ===========================================================================
# 参数校验
# ===========================================================================


@pytest.mark.parametrize("bad_input", [None, 123, b"model.epair", ["model.epair"]])
def test_constructor_rejects_invalid_type(bad_input):
    with pytest.raises(TypeError, match="epair_file must be"):
        EpairModelLoader(bad_input)


def test_constructor_rejects_empty_path():
    with pytest.raises(ValueError, match="non-empty"):
        EpairModelLoader("")


def test_constructor_accepts_pathlike(stub_library, monkeypatch):
    calls = []
    monkeypatch.setattr(model_loader, "_new_ge_graph", _fake_graph_factory(calls))
    with EpairModelLoader(pathlib.Path(EPAIR_PATH)) as loader:
        assert loader.get_ge_graph() is calls[-1]


# ===========================================================================
# 兼容性：backend so 缺失 / 缺 C ABI 符号（模拟老 whl）
# ===========================================================================


def test_load_failure_when_backend_so_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(torch_delegate, "cmake_prefix", lambda: str(tmp_path))
    _c_abi._load_backend_library.cache_clear()
    try:
        with pytest.raises(
            EpairModelLoaderError, match="Failed to load torch_delegate backend library"
        ):
            EpairModelLoader(EPAIR_PATH)
    finally:
        _c_abi._load_backend_library.cache_clear()


def test_load_failure_when_c_abi_symbols_missing(monkeypatch, tmp_path):
    libc_candidates = []
    for pattern in (
        "/lib/*/libc.so.6",
        "/usr/lib/*/libc.so.6",
        "/lib64/libc.so.6",
        "/usr/lib64/libc.so.6",
        "/lib/libc.so.6",
        "/usr/lib/libc.so.6",
    ):
        libc_candidates.extend(glob.glob(pattern))
    if not libc_candidates:
        pytest.skip("未找到可复用的无 C ABI 符号动态库（libc.so.6）")
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    shutil.copy2(libc_candidates[0], lib_dir / "libtorch_delegate_backend.so")
    monkeypatch.setattr(torch_delegate, "cmake_prefix", lambda: str(tmp_path))
    _c_abi._load_backend_library.cache_clear()
    try:
        with pytest.raises(
            EpairModelLoaderError, match="does not export the EpairModelLoader C ABI"
        ):
            EpairModelLoader(EPAIR_PATH)
    finally:
        _c_abi._load_backend_library.cache_clear()


# ===========================================================================
# GE 图绑定：_new_ge_graph 接入 CANN GraphEngine Python API（ge-py）
# ===========================================================================


def test_new_ge_graph_returns_live_graph_and_pointer():
    ge_graph_module = pytest.importorskip(
        "ge.graph", reason="CANN GraphEngine Python API (ge) 不可用"
    )
    graph, graph_ptr = model_loader._new_ge_graph()
    try:
        assert isinstance(graph, ge_graph_module.Graph)
        assert isinstance(graph_ptr, int)
        assert graph_ptr == graph._handle
        # 空图：未拷入内容前不含任何节点
        assert graph.get_all_nodes() == []
    finally:
        del graph  # 显式触发 ge-py 侧 DestroyGraph，便于及时释放


def test_new_ge_graph_wraps_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "ge.graph", None)
    with pytest.raises(EpairModelLoaderError, match="GraphEngine Python API"):
        model_loader._new_ge_graph()


def test_new_ge_graph_wraps_creation_error(monkeypatch):
    class _BrokenGraph:
        def __init__(self) -> None:
            raise RuntimeError("stub create failed")

    monkeypatch.setitem(
        sys.modules, "ge.graph", types.SimpleNamespace(Graph=_BrokenGraph)
    )
    with pytest.raises(
        EpairModelLoaderError, match="Failed to create an empty ge::Graph"
    ):
        model_loader._new_ge_graph()


def test_new_ge_graph_rejects_missing_native_handle(monkeypatch):
    class _GraphWithoutHandle:
        pass

    monkeypatch.setitem(
        sys.modules, "ge.graph", types.SimpleNamespace(Graph=_GraphWithoutHandle)
    )
    with pytest.raises(
        EpairModelLoaderError, match="does not expose the native handle"
    ):
        model_loader._new_ge_graph()


def test_new_ge_graph_rejects_empty_native_handle(monkeypatch):
    class _GraphWithEmptyHandle:
        _handle = 0

    monkeypatch.setitem(
        sys.modules, "ge.graph", types.SimpleNamespace(Graph=_GraphWithEmptyHandle)
    )
    with pytest.raises(EpairModelLoaderError, match="empty native handle"):
        model_loader._new_ge_graph()


# ===========================================================================
# 特性交叉：与 epair_load 并存调用互不干扰
# ===========================================================================


def test_coexists_with_epair_load(stub_library, monkeypatch):
    monkeypatch.setattr(model_loader, "_new_ge_graph", _fake_graph_factory([]))
    with EpairModelLoader(EPAIR_PATH) as loader:
        assert loader.get_ge_graph() is not None
        with pytest.raises(FileNotFoundError):
            torch_delegate.epair_load("/tmp/definitely_missing.epair")
        assert loader.get_ge_graph() is not None


# ===========================================================================
# 可靠性：错误串为线程局部（并发/跨线程互不串扰）
# ===========================================================================


def test_error_string_is_thread_local():
    library = ctypes.CDLL(STUB_SO)
    _c_abi._declare_abi(library)
    library.TdEpairModelLoaderLastError.argtypes = []
    library.TdEpairModelLoaderLastError.restype = ctypes.c_char_p
    library.TdEpairModelLoaderGetGraph.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.TdEpairModelLoaderGetGraph.restype = ctypes.c_int64

    os.environ["TD_STUB_GET_GRAPH_STATUS"] = "1"
    try:
        results = {}

        def failing_worker():
            results["failed_status"] = library.TdEpairModelLoaderGetGraph(None, None)
            results["failed_error"] = library.TdEpairModelLoaderLastError().decode()

        def clean_worker():
            # 全新线程从未出错：其线程局部错误串不受失败线程影响
            results["clean_error"] = library.TdEpairModelLoaderLastError().decode()

        failing = threading.Thread(target=failing_worker)
        failing.start()
        failing.join()
        clean = threading.Thread(target=clean_worker)
        clean.start()
        clean.join()
    finally:
        os.environ.pop("TD_STUB_GET_GRAPH_STATUS", None)

    assert results["failed_status"] == 1
    assert "stub error: GetGraph failed: status=FAILED" in results["failed_error"]
    assert results["clean_error"] == ""
