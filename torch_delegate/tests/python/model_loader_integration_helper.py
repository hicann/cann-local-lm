# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""端到端集成校验子进程：真实 backend so → 真实 GE 图 → CopyFrom。

由 test_model_loader_integration.py 调起；子进程内可预载 libasan，避免 ASAN 构建
的 backend 在 pytest 进程中 dlopen 时 abort。退出码约定：0=通过，1=校验失败，
2=环境不满足（父用例据此 skip）。
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ENV = 2


def _load_ge_api():
    try:
        import ge.graph  # noqa: F401
        from ge.offline_compile import build_finalize, build_initialize
    except (ImportError, OSError, RuntimeError) as exc:
        print(f"CANN GraphEngine Python API (ge) 不可用: {exc}", file=sys.stderr)
        return None
    return build_initialize, build_finalize


def _run_case(backend_so: str, fixture: str) -> int:
    from torch_delegate import _c_abi
    from torch_delegate.model_loader import EpairModelLoader, EpairModelLoaderError

    _c_abi._backend_library_path = lambda: backend_so
    _c_abi._load_backend_library.cache_clear()
    try:
        try:
            loader = EpairModelLoader(fixture)
        except EpairModelLoaderError as exc:
            message = str(exc)
            if "Failed to load torch_delegate backend library" in message:
                print(f"backend so 加载失败: {message}", file=sys.stderr)
                return EXIT_ENV
            print(f"EpairModelLoader 构造失败: {message}", file=sys.stderr)
            return EXIT_FAIL
        with loader:
            graph = loader.get_ge_graph()
        node_count = len(graph.get_all_nodes())
        if node_count != 6:
            print(f"节点数不匹配: expected=6, actual={node_count}", file=sys.stderr)
            return EXIT_FAIL
        return EXIT_OK
    except Exception as exc:  # pylint: disable=broad-except
        print(f"端到端校验失败: {exc}", file=sys.stderr)
        return EXIT_FAIL
    finally:
        _c_abi._load_backend_library.cache_clear()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-so", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--soc-version", default="Ascend910B")
    args = parser.parse_args()

    ge_api = _load_ge_api()
    if ge_api is None:
        return EXIT_ENV
    build_initialize, build_finalize = ge_api

    try:
        build_initialize({"ge.socVersion": args.soc_version, "ge.graphRunMode": "0"})
    except Exception as exc:  # pylint: disable=broad-except
        print(f"GE 初始化失败: {exc}", file=sys.stderr)
        return EXIT_ENV

    try:
        return _run_case(args.backend_so, args.fixture)
    finally:
        try:
            build_finalize()
        except Exception:  # pylint: disable=broad-except
            pass


if __name__ == "__main__":
    sys.exit(main())
