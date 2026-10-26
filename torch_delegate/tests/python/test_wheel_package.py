# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""wheel 打包内容校验。

直接读取 whl（zip）校验发布产物结构与元数据，不安装、不执行动态库。
默认取 ``output/package/`` 下最新的 whl，也可用环境变量 ``TORCH_DELEGATE_WHEEL``
指定；未构建 whl 时整组用例跳过。
"""

import os
import platform
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "torch_delegate"
EXPECTED_ARCH = {
    "x86_64": "linux_x86_64",
    "aarch64": "linux_aarch64",
    "arm64": "linux_aarch64",
}.get(platform.machine(), f"linux_{platform.machine()}")


def _find_wheel() -> Path | None:
    from_env = os.environ.get("TORCH_DELEGATE_WHEEL")
    if from_env:
        path = Path(from_env)
        return path if path.is_file() else None
    candidates = sorted((REPO_ROOT / "output" / "package").glob("torch_delegate-*.whl"))
    return candidates[-1] if candidates else None


WHEEL = _find_wheel()

pytestmark = pytest.mark.skipif(
    WHEEL is None,
    reason="wheel 未构建，请先执行 ./build.sh 或设置 TORCH_DELEGATE_WHEEL",
)

NAMES = zipfile.ZipFile(WHEEL).namelist() if WHEEL is not None else []
DIST_INFO = next((n for n in NAMES if n.endswith(".dist-info/METADATA")), "")


def _read(member_suffix: str) -> str:
    match = next(n for n in NAMES if n.endswith(member_suffix))
    with zipfile.ZipFile(WHEEL) as wheel_zip:
        return wheel_zip.read(match).decode("utf-8")


def test_wheel_filename_is_platform_tagged():
    assert re.fullmatch(
        rf"torch_delegate-.+-py3-none-{EXPECTED_ARCH}\.whl", WHEEL.name
    ), WHEEL.name


def test_python_api_packaged():
    for module in ("__init__.py", "api.py", "cli.py", "_version.py"):
        assert f"{PACKAGE}/{module}" in NAMES


def test_cpp_headers_packaged():
    assert f"{PACKAGE}/inc/{PACKAGE}/epair_model_loader.h" in NAMES
    assert f"{PACKAGE}/inc/{PACKAGE}/td_types.h" in NAMES


def test_shared_library_packaged():
    assert f"{PACKAGE}/lib/libtorch_delegate_backend.so" in NAMES


def test_cmake_config_packaged():
    prefix = f"{PACKAGE}/lib/cmake/TorchDelegate/"
    for name in (
        "TorchDelegateConfig.cmake",
        "TorchDelegateConfigVersion.cmake",
        "TorchDelegateTargets.cmake",
    ):
        assert prefix + name in NAMES
    assert any(n.startswith(prefix + "modules/") for n in NAMES)


def test_no_set_env_script():
    assert not any(n.endswith("set_env.sh") for n in NAMES)


def test_artifacts_at_wheel_root():
    assert not any(n.startswith(f"{PACKAGE}-") and ".data/" in n for n in NAMES)


def test_metadata_requires_python_and_declares_no_torch_dependency():
    metadata = _read(".dist-info/METADATA")
    requires_python = re.search(
        r"^Requires-Python: (.+)$", metadata, re.MULTILINE
    ).group(1)
    assert ">=3.10" in requires_python and "<3.14" in requires_python, requires_python
    assert not re.search(r"^Requires-Dist: torch([<>=! ]|$)", metadata, re.MULTILINE)
    assert "torch_npu" not in metadata and "torch-npu" not in metadata


def test_wheel_tag():
    wheel_meta = _read(".dist-info/WHEEL")
    assert f"Tag: py3-none-{EXPECTED_ARCH}" in wheel_meta


def test_console_script_entry_point():
    assert "torch-delegate-config = torch_delegate.cli:main" in _read(
        ".dist-info/entry_points.txt"
    )


def test_version_matches_metadata():
    version = re.search(
        r"^Version: (.+)$", _read(".dist-info/METADATA"), re.MULTILINE
    ).group(1)
    assert f'__version__ = "{version}"' in _read(f"{PACKAGE}/_version.py")


def test_shared_library_has_no_torch_or_cann_runpath():
    readelf = subprocess.run(["which", "readelf"], capture_output=True)
    if readelf.returncode != 0:
        pytest.skip("readelf 不可用")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(WHEEL) as wheel_zip:
            wheel_zip.extract(f"{PACKAGE}/lib/libtorch_delegate_backend.so", tmp)
        output = subprocess.run(
            [
                "readelf",
                "-d",
                str(Path(tmp) / PACKAGE / "lib" / "libtorch_delegate_backend.so"),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    assert "RPATH" not in output and "RUNPATH" not in output, output
