# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""libtorch_delegate_backend.so 的内部 C ABI ctypes 绑定。

对应头文件 ``csrc/backend/epair_model_loader_c.h``（backend 内部实现，非对外接口，
不随 whl 发布、不承诺兼容性）。仅使用标准库 ctypes，签名与头文件一一对应；
状态码到 Python 异常的映射收敛在本模块，供 :mod:`torch_delegate.model_loader` 使用。
"""

from __future__ import annotations

import ctypes
import functools
import os

from .core.utils import logger

__all__ = ["EpairModelLoaderError", "create_loader", "destroy_loader", "get_graph"]

# 状态码定义与 td_types.h / epair_model_loader_c.h 保持一致。
SUCCESS = 0
FAILED = 1
INVALID_PARAM = 2
INTERNAL_ERROR = 3

_STATUS_NAMES = {
    SUCCESS: "SUCCESS",
    FAILED: "FAILED",
    INVALID_PARAM: "INVALID_PARAM",
    INTERNAL_ERROR: "INTERNAL_ERROR",
}

_BACKEND_LIBRARY_NAME = "libtorch_delegate_backend.so"

_LOAD_FAILURE_HINT = (
    "Check that the torch_delegate wheel is fully installed (the library lives at "
    "torch_delegate/lib/libtorch_delegate_backend.so) and that LD_LIBRARY_PATH covers "
    "libtorch and CANN runtime libraries per the build guide."
)


class EpairModelLoaderError(RuntimeError):
    """EpairModelLoader C ABI 调用失败（状态码非 0）或 backend 加载失败。"""


def _backend_library_path() -> str:
    """由 torch_delegate.cmake_prefix() 推导 backend so 路径，不硬编码安装位置。

    函数内导入以规避包初始化期的循环导入。
    """
    from . import cmake_prefix

    return os.path.join(cmake_prefix(), "lib", _BACKEND_LIBRARY_NAME)


@functools.lru_cache(maxsize=1)
def _load_backend_library() -> ctypes.CDLL:
    """进程级加载并解析 backend so，仅执行一次。"""
    library_path = _backend_library_path()
    try:
        library = ctypes.CDLL(library_path)
        _declare_abi(library)
    except OSError as exc:
        raise EpairModelLoaderError(
            f"Failed to load torch_delegate backend library '{library_path}': {exc}. {_LOAD_FAILURE_HINT}"
        ) from exc
    except AttributeError as exc:
        raise EpairModelLoaderError(
            f"torch_delegate backend library '{library_path}' does not export the "
            f"EpairModelLoader C ABI ({exc}); the installed wheel is outdated. "
            "Please reinstall a matching torch_delegate version."
        ) from exc
    return library


def _declare_abi(library: ctypes.CDLL) -> None:
    """显式声明 C ABI 四函数签名，与内部头文件 epair_model_loader_c.h 一一对应。"""
    library.TdEpairModelLoaderCreate.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.TdEpairModelLoaderCreate.restype = ctypes.c_int64

    library.TdEpairModelLoaderGetGraph.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.TdEpairModelLoaderGetGraph.restype = ctypes.c_int64

    library.TdEpairModelLoaderDestroy.argtypes = [ctypes.c_void_p]
    library.TdEpairModelLoaderDestroy.restype = ctypes.c_int64

    library.TdEpairModelLoaderLastError.argtypes = []
    library.TdEpairModelLoaderLastError.restype = ctypes.c_char_p


def _status_name(status: int) -> str:
    return _STATUS_NAMES.get(status, f"UNKNOWN_STATUS({status})")


def _last_error() -> str:
    raw = _load_backend_library().TdEpairModelLoaderLastError()
    return raw.decode("utf-8", errors="replace") if raw else ""


def _raise_for_status(operation: str, status: int) -> None:
    if status == SUCCESS:
        return
    message = f"EpairModelLoader {operation} failed: status={_status_name(status)}"
    detail = _last_error()
    if detail:
        message = f"{message}, detail={detail}"
    if status == INVALID_PARAM:
        raise ValueError(message)
    raise EpairModelLoaderError(message)


def create_loader(archive_path: str) -> int:
    """创建 C++ loader（解析归档并构建 GE 图），返回不透明句柄地址。"""
    handle = ctypes.c_void_p()
    status = _load_backend_library().TdEpairModelLoaderCreate(
        archive_path.encode("utf-8"), ctypes.byref(handle)
    )
    _raise_for_status("create", status)
    return handle.value


def get_graph(handle: int, graph_ptr: int) -> None:
    """将内部 GE 图拷贝至 graph_ptr 指向的 ge::Graph 对象（独立副本）。"""
    status = _load_backend_library().TdEpairModelLoaderGetGraph(handle, graph_ptr)
    _raise_for_status("get_ge_graph", status)


def destroy_loader(handle: int) -> None:
    """销毁 C++ loader；NULL 安全，失败仅记录日志（不抛出，避免析构路径二次异常）。

    同一非空句柄只能销毁一次；调用方（``EpairModelLoader.close``）保证不会重复销毁。
    """
    try:
        status = _load_backend_library().TdEpairModelLoaderDestroy(handle)
        _raise_for_status("close", status)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("EpairModelLoader close failed: %s", exc)
