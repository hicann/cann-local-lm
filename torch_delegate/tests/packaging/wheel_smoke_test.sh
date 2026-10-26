#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# wheel_smoke_test.sh - torch_delegate whl 发布产物端到端冒烟测试
#
# 用法:
#   bash tests/packaging/wheel_smoke_test.sh [wheel_path] [--python=<interpreter>]
#
# 步骤:
#   1. 定位 whl（参数 / TORCH_DELEGATE_WHEEL / output/package 下最新产物）
#   2. 创建临时 venv（--system-site-packages 以复用已装 torch）并 pip install --no-deps
#   3. 校验 import torch_delegate、__version__、cmake_prefix()
#   4. 校验 torch-delegate-config --cmake-prefix / --version
#   5. 校验动态库不含指向 torch/CANN 的 RPATH/RUNPATH
#   6. 若 CANN 与 cmake 可用，生成最小 C++ 消费工程做 find_package + 编译 + 运行
#
# 说明: 本脚本只验证打包与接入链路，不需要 NPU 设备。为避开宿主环境中
#       torch_npu 自动加载失败，显式设置 TORCH_DEVICE_BACKEND_AUTOLOAD=0。

set -euo pipefail

export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WHEEL_PATH="${TORCH_DELEGATE_WHEEL:-}"
PYTHON_BIN=""
WORK_DIR=""

log_step() { echo ""; echo "=== $1 ==="; }
log_ok() { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1" >&2; exit 1; }

# 在中立目录执行：CANN 的 set_env.sh 导出的 PYTHONPATH 带尾随冒号，空条目会被
# Python 当作当前工作目录，导致在仓库根目录执行时源码树遮蔽已安装的包
venv_py() { (cd "${WORK_DIR}" && "${WORK_DIR}/venv/bin/python" "$@"); }
venv_cli() { (cd "${WORK_DIR}" && "${WORK_DIR}/venv/bin/torch-delegate-config" "$@"); }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --python=*) PYTHON_BIN="${1#*=}"; shift ;;
            --help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
            *) WHEEL_PATH="$1"; shift ;;
        esac
    done
    if [[ -z "${PYTHON_BIN}" ]]; then
        PYTHON_BIN="${REPO_ROOT}/.venv/bin/python3"
        [[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN="$(command -v python3)"
    fi
}

locate_wheel() {
    log_step "定位 whl"
    if [[ -z "${WHEEL_PATH}" ]]; then
        WHEEL_PATH="$(ls -t "${REPO_ROOT}"/output/package/torch_delegate-*.whl 2>/dev/null | head -1 || true)"
    fi
    [[ -n "${WHEEL_PATH}" && -f "${WHEEL_PATH}" ]] || fail "未找到 whl，请先执行 ./build.sh"
    log_ok "$(basename "${WHEEL_PATH}")"
    case "$(basename "${WHEEL_PATH}")" in
        *-py3-none-linux_*.whl) log_ok "平台 tag 正确" ;;
        *) fail "whl 文件名不符合 py3-none-linux_<arch> 规范" ;;
    esac
}

install_into_venv() {
    log_step "安装到临时 venv"
    WORK_DIR="$(mktemp -d /tmp/td_wheel_smoke.XXXXXX)"
    "${PYTHON_BIN}" -m venv --system-site-packages "${WORK_DIR}/venv"
    "${WORK_DIR}/venv/bin/pip" install -q --no-deps --force-reinstall "${WHEEL_PATH}"
    log_ok "pip install 完成"
}

check_python_api() {
    log_step "校验 Python API 与 CLI"
    local prefix version
    prefix="$(venv_py -c 'import torch_delegate; print(torch_delegate.cmake_prefix())')"
    version="$(venv_py -c 'import torch_delegate; print(torch_delegate.__version__)')"
    [[ -d "${prefix}/inc" && -d "${prefix}/lib" ]] || fail "前缀目录缺少 inc/ 或 lib/: ${prefix}"
    log_ok "cmake_prefix()=${prefix}"
    log_ok "__version__=${version}"

    local cli_prefix cli_version
    cli_prefix="$(venv_cli --cmake-prefix)"
    cli_version="$(venv_cli --version)"
    [[ "${cli_prefix}" == "${prefix}" ]] || fail "CLI 前缀与 API 不一致: ${cli_prefix} != ${prefix}"
    [[ "${cli_version}" == "${version}" ]] || fail "CLI 版本与 API 不一致"
    log_ok "torch-delegate-config --cmake-prefix/--version 一致"
}

check_no_runpath() {
    log_step "校验动态库无 torch/CANN RUNPATH"
    command -v readelf >/dev/null || { echo "  [SKIP] readelf 不可用"; return 0; }
    local so="${WORK_DIR}/venv/lib/python*/site-packages/${1:-torch_delegate}/lib/libtorch_delegate_backend.so"
    local so_path
    so_path="$(echo ${so} | awk '{print $1}')"
    [[ -f "${so_path}" ]] || fail "未找到 ${so_path}"
    if readelf -d "${so_path}" | grep -Eq "RPATH|RUNPATH"; then
        readelf -d "${so_path}" | grep -E "RPATH|RUNPATH"
        fail "动态库存在 RPATH/RUNPATH"
    fi
    log_ok "无 RPATH/RUNPATH（运行期依赖由用户配置 LD_LIBRARY_PATH）"
}

check_cmake_consumer() {
    log_step "校验 C++ 消费工程 find_package"
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        echo "  [SKIP] ASCEND_HOME_PATH 未设置"
        return 0
    fi
    command -v cmake >/dev/null || { echo "  [SKIP] cmake 不可用"; return 0; }

    local prefix torch_pkg_dir proj
    prefix="$(venv_cli --cmake-prefix 2>/dev/null || true)"
    [[ -n "${prefix}" ]] || prefix="$(venv_py -c 'import torch_delegate; print(torch_delegate.cmake_prefix())')"
    torch_pkg_dir="$(venv_py -c 'import torch, os; print(os.path.dirname(torch.__file__))')" || {
        echo "  [SKIP] 无法获取 torch 包目录"; return 0;
    }

    proj="${WORK_DIR}/consumer"
    mkdir -p "${proj}"
    cat > "${proj}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.18)
project(td_smoke_consumer CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORCH_PKG_DIR "" CACHE PATH "Path to Python torch package directory")
set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}")
find_package(TorchDelegate REQUIRED)
set(TORCH_LINK_DIRS "${TORCH_PKG_DIR}/lib")
get_filename_component(_torch_parent "${TORCH_PKG_DIR}" DIRECTORY)
if(EXISTS "${_torch_parent}/torch.libs")
    list(APPEND TORCH_LINK_DIRS "${_torch_parent}/torch.libs")
endif()
add_executable(app main.cpp)
target_include_directories(app PRIVATE ${CANN_ROOT}/include)
target_link_directories(app PRIVATE ${CANN_ROOT}/lib64 ${TORCH_LINK_DIRS})
foreach(_d IN LISTS TORCH_LINK_DIRS)
    target_link_options(app PRIVATE "LINKER:-rpath-link,${_d}")
endforeach()
target_link_options(app PRIVATE "LINKER:-rpath-link,${CANN_ROOT}/lib64")
target_link_libraries(app PRIVATE TorchDelegate::torch_delegate_backend ge_runner_v2 acl_rt)
EOF
    cat > "${proj}/main.cpp" <<'EOF'
#include <cstdio>

#include "torch_delegate/epair_model_loader.h"
#include "torch_delegate/td_types.h"

int main() {
  std::printf("smoke ok, td::SUCCESS=%d\n", static_cast<int>(td::SUCCESS));
  return 0;
}
EOF

    cmake -S "${proj}" -B "${proj}/build" \
        -DCMAKE_PREFIX_PATH="${prefix}" \
        -DTORCH_PKG_DIR="${torch_pkg_dir}" > "${WORK_DIR}/cmake_configure.log" 2>&1 \
        || { tail -20 "${WORK_DIR}/cmake_configure.log"; fail "find_package(TorchDelegate) 配置失败"; }
    log_ok "find_package 配置成功（前缀来自 whl）"

    cmake --build "${proj}/build" -j"$(nproc 2>/dev/null || echo 4)" > "${WORK_DIR}/cmake_build.log" 2>&1 \
        || { tail -20 "${WORK_DIR}/cmake_build.log"; fail "C++ 消费工程编译失败"; }
    log_ok "编译成功"

    local ld_path="${torch_pkg_dir}/lib:${torch_pkg_dir}.libs:${prefix}/lib:${ASCEND_HOME_PATH}/lib64"
    LD_LIBRARY_PATH="${ld_path}:${LD_LIBRARY_PATH:-}" "${proj}/build/app" || fail "运行失败"
    log_ok "运行成功（LD_LIBRARY_PATH 手动配置）"
}

cleanup() {
    [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]] && rm -rf "${WORK_DIR}"
}

main() {
    parse_args "$@"
    locate_wheel
    install_into_venv
    trap cleanup EXIT
    check_python_api
    check_no_runpath
    check_cmake_consumer
    echo ""
    echo "=== 冒烟测试通过 ==="
}

main "$@"
