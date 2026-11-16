# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""EpairModelLoader 端到端集成用例：真实 backend so → 真实 GE 图 → CopyFrom。

依赖 build_test.sh 产出的 backend so 与 annotated_mm_add.epair fixture，默认路径
自动发现；环境不满足时跳过。

build_test.sh 默认以 ASAN 构建 backend，Python 进程直接 dlopen 会因 ASAN 运行时
未先加载而 abort，因此实际校验放到子进程执行，并在子进程内通过 LD_PRELOAD 预载
libasan；门禁脚本无需为本用例做特殊处理。
"""

from __future__ import annotations

import ctypes.util
import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_BACKEND_SO = REPO_ROOT / "build_tests" / "csrc" / "libtorch_delegate_backend.so"
DEFAULT_FIXTURE = (
    REPO_ROOT / "build_tests" / "tests" / "cpp" / "fixtures" / "annotated_mm_add.epair"
)
BACKEND_SO_ENV = "TORCH_DELEGATE_BACKEND_SO"
FIXTURE_ENV = "TORCH_DELEGATE_EPAIR_FIXTURE"
HELPER = pathlib.Path(__file__).with_name("model_loader_integration_helper.py")


def _from_env_or_default(env_name: str, default: pathlib.Path) -> pathlib.Path:
    from_env = os.environ.get(env_name)
    return pathlib.Path(from_env) if from_env else default


def _find_asan_runtime() -> str | None:
    try:
        result = subprocess.run(
            ["gcc", "-print-file-name=libasan.so"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        candidate = result.stdout.strip()
        if candidate and pathlib.Path(candidate).is_file():
            return candidate
    return ctypes.util.find_library("asan")


def test_real_backend_get_ge_graph_end_to_end():
    backend_so = _from_env_or_default(BACKEND_SO_ENV, DEFAULT_BACKEND_SO)
    fixture = _from_env_or_default(FIXTURE_ENV, DEFAULT_FIXTURE)
    if not backend_so.is_file() or not fixture.is_file():
        pytest.skip("真实 backend so 或 annotated_mm_add.epair 未构建，跳过端到端用例")

    env = os.environ.copy()
    asan_runtime = _find_asan_runtime()
    if asan_runtime:
        preload = env.get("LD_PRELOAD")
        env["LD_PRELOAD"] = f"{asan_runtime}:{preload}" if preload else asan_runtime
        env["ASAN_OPTIONS"] = "detect_leaks=0"

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(HELPER),
                "--backend-so",
                str(backend_so),
                "--fixture",
                str(fixture),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"端到端集成用例超时: {exc}")

    if proc.returncode == 2:
        pytest.skip(proc.stderr.strip() or "端到端集成用例环境不满足")
    if proc.returncode != 0:
        if "ASan runtime does not come first" in proc.stderr:
            pytest.skip("ASAN 构建的 backend 需要预载 libasan，当前环境无法预载")
        pytest.fail(
            f"端到端集成用例失败 (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
