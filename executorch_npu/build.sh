#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Build the ExecuTorch-NPU backend archives and platform wheel.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="Release"

usage() {
    cat <<EOF
Usage: $0 [--debug] [--help]

  --debug  Build with CMAKE_BUILD_TYPE=Debug (default: Release)
  --help   Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug)
            BUILD_TYPE="Debug"
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

BUILD_DIR="${PROJECT_ROOT}/build/${BUILD_TYPE,,}"
OUTPUT_DIR="${PROJECT_ROOT}/output/package"

phase() {
    printf '\n==> %s\n' "$1"
}

phase "Check build environment"

PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Python not found. Activate a Python 3.10+ environment." >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c '
import sys
assert sys.version_info >= (3, 10)
import packaging
import setuptools
import wheel
' >/dev/null 2>&1; then
    echo "Python 3.10+ with setuptools, wheel and packaging is required." >&2
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "CMake 3.24+ is required." >&2
    exit 1
fi

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "ASCEND_HOME_PATH is not set. Source the CANN set_env.sh first." >&2
    exit 1
fi

EXECUTORCH_ROOT="$(${PYTHON_BIN} - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import site
import sys
from packaging.version import Version

try:
    installed_version = Version(version("executorch"))
except PackageNotFoundError as exc:
    raise SystemExit("ExecuTorch 1.5.0 full wheel is not installed") from exc

if installed_version < Version("1.5.0"):
    raise SystemExit(
        f"ExecuTorch {installed_version} is unsupported; install 1.5.0 or newer"
    )
if installed_version >= Version("1.6.0"):
    print(
        f"Warning: ExecuTorch {installed_version} is newer than the supported 1.5 release line",
        file=sys.stderr,
    )

for site_dir in site.getsitepackages():
    root = Path(site_dir) / "executorch"
    required = (
        root / "include/executorch/runtime/backend/interface.h",
        root / "include/executorch/extension/module/module.h",
        root / "lib/libexecutorch.so",
        root / "share/cmake/executorch-config.cmake",
    )
    if all(path.is_file() for path in required):
        print(root.resolve())
        break
else:
    raise SystemExit("ExecuTorch 1.5.0 full wheel with the C++ SDK is not installed")
PY
)"
EXECUTORCH_INCLUDE_DIR="${EXECUTORCH_ROOT}/include"

printf 'Python:      %s\n' "$("${PYTHON_BIN}" --version 2>&1)"
printf 'CMake:       %s\n' "$(cmake --version | head -1)"
printf 'CANN:        %s\n' "${ASCEND_HOME_PATH}"
printf 'ExecuTorch:  %s\n' "${EXECUTORCH_ROOT}"

phase "Configure and build C++ backend"

cmake -E remove_directory "${BUILD_DIR}/install"
cmake --fresh \
    -S "${PROJECT_ROOT}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/install" \
    -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
    -DEXECUTORCH_INCLUDE_DIR="${EXECUTORCH_INCLUDE_DIR}" \
    -Dexecutorch_DIR="${EXECUTORCH_ROOT}/share/cmake"

cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || echo 4)"
cmake --install "${BUILD_DIR}"

phase "Run C++ tests"
ctest --test-dir "${BUILD_DIR}" --output-on-failure

phase "Build Python wheel"
mkdir -p "${OUTPUT_DIR}"

EXECUTORCH_NPU_PACKAGE_SDK=1 \
EXECUTORCH_NPU_SDK_INSTALL_ROOT="${BUILD_DIR}/install" \
EXECUTORCH_NPU_PACKAGE_BUILD_TYPE="${BUILD_TYPE}" \
EXECUTORCH_NPU_EXECUTORCH_ROOT="${EXECUTORCH_ROOT}" \
EXECUTORCH_NPU_CANN_ROOT="${ASCEND_HOME_PATH}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/setup.py" bdist_wheel \
        --dist-dir "${OUTPUT_DIR}"

WHEEL_PATH="$(ls -t "${OUTPUT_DIR}"/executorch_npu-*.whl | head -1)"

printf '\nBuild completed.\n'
printf '  Build type: %s\n' "${BUILD_TYPE}"
printf '  Install:    %s\n' "${BUILD_DIR}/install"
printf '  Wheel:      %s\n' "${WHEEL_PATH}"
