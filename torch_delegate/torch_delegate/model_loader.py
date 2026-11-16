# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""EpairModelLoader：加载 ``.epair`` 归档并构建 GE 图，供 GE Offline_compile 使用。

典型用法（executortorch delegate_adapter 对接方式；构造 loader 前须先完成 GE
初始化，否则 C ABI 返回 ``FAILED``）::

    ge.offline_compile.build_initialize({"ge.socVersion": "Ascend910B", "ge.graphRunMode": "0"})
    try:
        with torch_delegate.EpairModelLoader(epair_path) as loader:
            ge_graph = loader.get_ge_graph()
    finally:
        ge.offline_compile.build_finalize()
"""

from __future__ import annotations

import os
from typing import Any

from . import _c_abi
from ._c_abi import EpairModelLoaderError

__all__ = ["EpairModelLoader", "EpairModelLoaderError"]


def _new_ge_graph() -> tuple[Any, int]:
    """创建空 ``ge::Graph`` 并返回 ``(图对象, 本机指针)``。

    依赖 CANN 随包发布的 GraphEngine Python API（``ge.graph.Graph``，提供空图
    创建与原生指针访问）；指针交由 backend C ABI 拷贝填充图内容，图对象生命周期
    由返回对象自行管理。
    """
    try:
        from ge.graph import Graph
    except (ImportError, OSError, RuntimeError) as exc:
        raise EpairModelLoaderError(
            "Creating an empty ge::Graph requires the CANN GraphEngine Python API "
            "(module 'ge', shipped with CANN under python/site-packages), which is "
            f"not available in the current environment: {exc}"
        ) from exc
    try:
        graph = Graph()
    except RuntimeError as exc:
        raise EpairModelLoaderError(
            f"Failed to create an empty ge::Graph via ge.graph.Graph: {exc}"
        ) from exc
    try:
        graph_ptr = graph._handle
    except AttributeError as exc:
        raise EpairModelLoaderError(
            "ge.graph.Graph does not expose the native handle (_handle); "
            "the installed CANN GraphEngine Python API is incompatible"
        ) from exc
    if not graph_ptr:
        raise EpairModelLoaderError("ge.graph.Graph returned an empty native handle")
    return graph, graph_ptr


class EpairModelLoader:
    """从 ``.epair`` 归档构建 GE 图的 Python 加载器。

    构造时通过 C ABI 创建 C++ ``td::EpairModelLoader`` 并完成归档解析与
    GE 图构建（不编译）；:meth:`get_ge_graph` 每次返回独立持有的 GE 图副本。
    实例生命周期由本对象独占管理（构造/``close()``/析构三态），同一实例
    不承诺多线程并发安全（与 C++ 侧约定一致）。
    """

    def __init__(self, epair_file: str | os.PathLike[str]) -> None:
        """加载 ``.epair`` 归档并构建 GE 图。

        Args:
            epair_file: ``.epair`` 归档文件路径（绝对/相对均可）。

        Raises:
            TypeError: ``epair_file`` 不是 ``str`` 或 ``os.PathLike[str]``。
            ValueError: ``epair_file`` 为空路径。
            EpairModelLoaderError: backend so 加载失败，或归档解析/图构建失败
                （GE 未初始化时通常也会以该异常返回）。
        """
        self._handle: int | None = None
        self._closed = False
        self._handle = _c_abi.create_loader(self._normalize_path(epair_file))

    @staticmethod
    def _normalize_path(epair_file: str | os.PathLike[str]) -> str:
        if not isinstance(epair_file, (str, os.PathLike)):
            raise TypeError(
                f"epair_file must be str or os.PathLike[str], got {type(epair_file).__name__}"
            )
        path = os.fspath(epair_file)
        if not path:
            raise ValueError("epair_file must be a non-empty path")
        return path

    def get_ge_graph(self) -> Any:
        """获取内部 GE 图的独立副本，可重复调用，副本间互不影响。

        Returns:
            GE 图对象（CANN GraphEngine Python API 的 ``ge.graph.Graph``），
            可直接交由 ``ge.offline_compile.build_model`` 离线编译消费。

        Raises:
            EpairModelLoaderError: 实例已 ``close``、GraphEngine Python API
                不可用，或 C++ 侧图拷贝失败。
        """
        if self._closed:
            raise EpairModelLoaderError(
                "EpairModelLoader is closed; get_ge_graph() requires a live loader"
            )
        graph, graph_ptr = _new_ge_graph()
        _c_abi.get_graph(self._handle, graph_ptr)
        return graph

    def close(self) -> None:
        """显式销毁 C++ loader；重复调用幂等（no-op）。"""
        if self._closed:
            return
        self._closed = True
        if self._handle is not None:
            handle, self._handle = self._handle, None
            _c_abi.destroy_loader(handle)

    def __enter__(self) -> "EpairModelLoader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # 析构兜底释放，保证无泄漏；解释器退出阶段不抛异常。
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass
