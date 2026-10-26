# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TorchDelegate wheel 构建脚本。

把 CMake 安装树（对外头文件、动态库、CMake 配置包）与 Python API 打进同一个
平台标签化 wheel。构建期输入由环境变量注入：

    TORCH_DELEGATE_CPP_INSTALL_DIR   CMake 安装树根目录（必需，含 inc/ 与 lib/）
    TORCH_DELEGATE_PACKAGE_VERSION   版本号（缺省时回退解析 build.conf）

wheel 的 Python/ABI tag 固定为 ``py3-none``（动态库不链接 libpython），平台 tag
由 :func:`detect_platform` 推导，因此同一 wheel 可安装进 3.10 ~ 3.13。
"""

import os
import platform
import shutil
import sys

from setuptools import setup
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "torch_delegate"
CPP_INSTALL_ENV = "TORCH_DELEGATE_CPP_INSTALL_DIR"
VERSION_ENV = "TORCH_DELEGATE_PACKAGE_VERSION"
FALLBACK_VERSION = "0.0.0.dev0"
STAGED_SUBDIRS = ("inc", "lib")


def detect_platform() -> str:
    """把 ``platform.machine()`` 映射为 wheel 平台 tag。"""
    machine = platform.machine()
    if machine == "x86_64":
        return "linux_x86_64"
    if machine in ("aarch64", "arm64"):
        return "linux_aarch64"
    return f"linux_{machine}"


def _read_build_conf_version() -> str:
    conf_path = os.path.join(PROJECT_ROOT, "build.conf")
    try:
        with open(conf_path, encoding="utf-8") as conf:
            for line in conf:
                stripped = line.strip()
                if stripped.startswith("TORCH_DELEGATE_VERSION="):
                    return stripped.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return FALLBACK_VERSION


def resolve_version() -> str:
    """版本号单一来源：环境变量优先，其次 build.conf。"""
    return os.environ.get(VERSION_ENV) or _read_build_conf_version()


def _fail(message: str) -> None:
    print(f"[torch-delegate] {message}", file=sys.stderr)
    raise SystemExit(1)


def _require_cpp_install_dir() -> str:
    install_dir = os.environ.get(CPP_INSTALL_ENV, "")
    if not install_dir:
        _fail(
            f"{CPP_INSTALL_ENV} is not set; cannot embed C++ artifacts into the wheel.\n"
            "  Run ./build.sh, or build and install manually then export:\n"
            f"    export {CPP_INSTALL_ENV}=$PWD/build/release/install"
        )
    missing = [
        d for d in STAGED_SUBDIRS if not os.path.isdir(os.path.join(install_dir, d))
    ]
    if missing:
        _fail(
            f"CMake install tree is incomplete: {install_dir} (missing: {', '.join(missing)}).\n"
            "  Run 'cmake --install <build-dir> --prefix <install-dir>' first."
        )
    return install_dir


def _stage_cpp_artifacts(build_lib: str) -> None:
    install_dir = _require_cpp_install_dir()
    package_dir = os.path.join(build_lib, PACKAGE_NAME)
    for sub in STAGED_SUBDIRS:
        shutil.copytree(
            os.path.join(install_dir, sub),
            os.path.join(package_dir, sub),
            symlinks=True,
            dirs_exist_ok=True,
        )


def _write_version_file(build_lib: str, version: str) -> None:
    package_dir = os.path.join(build_lib, PACKAGE_NAME)
    os.makedirs(package_dir, exist_ok=True)
    version_path = os.path.join(package_dir, "_version.py")
    with open(version_path, "w", encoding="utf-8") as version_file:
        version_file.write(f'__version__ = "{version}"\n')


class BuildPy(build_py):
    """在常规 Python 打包之外，注入 C++ 制品与构建期版本号。"""

    def run(self) -> None:
        super().run()
        _stage_cpp_artifacts(self.build_lib)
        _write_version_file(self.build_lib, resolve_version())


class BdistWheel(bdist_wheel):
    """产出 ``py3-none-linux_<arch>`` 平台 wheel。

    包内无 C 扩展，故保持 ``root_is_pure=True``（包内容置于 wheel 根，安装到
    site-packages），仅重写 tag 以声明平台相关性；与 ``nvidia-*`` 等
    ``py3-none-<platform>`` wheel 的做法一致。若改为 ``root_is_pure=False``，
    setuptools 会把全部内容挪进 ``<dist>.data/purelib/``，破坏 PyTorch 式布局。
    """

    def get_tag(self):
        return ("py3", "none", detect_platform())


setup(
    version=resolve_version(),
    cmdclass={"build_py": BuildPy, "bdist_wheel": BdistWheel},
)
