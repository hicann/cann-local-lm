# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


ROOT = Path(__file__).resolve().parents[1]


def _assignment(text: str, name: str, default: str = "unknown") -> str:
    for line in text.splitlines():
        if line.startswith(name) and "=" in line:
            return line.split("=", 1)[1].strip().strip("\"'")
    return default


def _executorch_identity() -> tuple[str, str]:
    version_file = Path(os.environ["EXECUTORCH_NPU_EXECUTORCH_ROOT"]) / "version.py"
    text = version_file.read_text(encoding="utf-8")
    return _assignment(text, "__version__"), _assignment(text, "git_version")


def _cann_version() -> str:
    root = Path(os.environ["EXECUTORCH_NPU_CANN_ROOT"])
    candidates = (
        root / "version.info",
        root / "opp" / "version.info",
        root.parent / "version.info",
        root.parent / "opp" / "version.info",
        root / "ascend_toolkit_install.info",
        root.parent / "ascend_toolkit_install.info",
    )
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for key in ("Version", "version", "version_dir"):
            value = _assignment(text, key, default="")
            if value:
                return value
    return "unknown"


def _compiler_version() -> str:
    try:
        return subprocess.check_output(
            ["c++", "-dumpfullversion", "-dumpversion"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class BuildPyWithSdk(build_py):
    def run(self) -> None:
        super().run()
        if os.environ.get("EXECUTORCH_NPU_PACKAGE_SDK") != "1":
            return

        install_root = Path(os.environ["EXECUTORCH_NPU_SDK_INSTALL_ROOT"]).resolve()
        required = (
            install_root / "lib" / "libexecutorch_npu_manifest.a",
            install_root / "lib" / "libexecutorch_npu_backend.a",
            install_root / "include" / "executorch_npu" / "state.h",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                "ExecuTorch-NPU SDK install is incomplete: " + ", ".join(missing)
            )

        sdk_root = Path(self.build_lib) / "executorch_npu" / "_sdk"
        if sdk_root.exists():
            shutil.rmtree(sdk_root)
        sdk_root.mkdir(parents=True)

        (sdk_root / "cmake").mkdir()
        (sdk_root / "include" / "executorch_npu").mkdir(parents=True)
        (sdk_root / "lib").mkdir()
        for cmake_file in (
            "executorch_npu_backend.cmake",
            "executorch_npu_executorch.cmake",
        ):
            shutil.copy2(ROOT / "cmake" / cmake_file, sdk_root / "cmake")
        shutil.copy2(
            ROOT / "runtime" / "include" / "executorch_npu" / "state.h",
            sdk_root / "include" / "executorch_npu",
        )
        for library in required[:2]:
            shutil.copy2(library, sdk_root / "lib")
        executorch_version, executorch_commit = _executorch_identity()
        compatibility = {
            "package_version": self.distribution.get_version(),
            "build_type": os.environ.get(
                "EXECUTORCH_NPU_PACKAGE_BUILD_TYPE", "unknown"
            ),
            "built_with": {
                "python": platform.python_version(),
                "torch": "not-required",
                "executorch": executorch_version,
                "executorch_commit": executorch_commit,
                "cann": _cann_version(),
                "compiler": _compiler_version(),
            },
        }
        (sdk_root / "compatibility.json").write_text(
            json.dumps(compatibility, indent=2) + "\n", encoding="utf-8"
        )


class PlatformWheel(bdist_wheel):
    def finalize_options(self) -> None:
        super().finalize_options()
        if os.environ.get("EXECUTORCH_NPU_PACKAGE_SDK") == "1":
            self.root_is_pure = False


setup(cmdclass={"build_py": BuildPyWithSdk, "bdist_wheel": PlatformWheel})
