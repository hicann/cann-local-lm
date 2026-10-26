#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# run.sh - _assert_tensor_metadata demo 一键验证脚本
#
# 流程:
#   [1] 运行 export_demo.py (导出 epair + 保存输入/参考输出)
#   [2] 构建 C++ EpairModelLoader
#   [3] 运行 C++ 加载器, 执行 epair 推理
#
# 前置条件:
#   1. source <cann_path>/set_env.sh
#   2. python3 环境已激活, torch_npu/torch_delegate(whl) 已安装
#      TORCH_DELEGATE_ROOT 会自动从已安装的 whl 推导, 无需 source set_env.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHASE=0
TOTAL=3

phase() {
    PHASE=$((PHASE + 1))
    echo ""
    echo "========================================"
    echo " [${PHASE}/${TOTAL}] $1"
    echo "========================================"
}

check() { echo "  [OK] $1"; }
fail() { echo "  [FAIL] $1"; exit 1; }
skip() { echo "  [SKIP] $1"; }

# ---- [1/3] 导出 epair ----
step1_export() {
    phase "导出 epair + 保存输入/参考输出"
    python3 "$SCRIPT_DIR/export_demo.py"
    check "epair 导出完成"
}

# ---- [2/3] 构建 C++ EpairModelLoader ----
step2_build_cpp() {
    phase "构建 C++ EpairModelLoader"

    if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
        TORCH_DELEGATE_ROOT="$(python3 -c 'import torch_delegate; print(torch_delegate.cmake_prefix())' 2>/dev/null || true)"
        export TORCH_DELEGATE_ROOT
    fi
    if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
        fail "TORCH_DELEGATE_ROOT 未设置, 且当前 Python 未安装 torch_delegate; 请先 pip install <torch_delegate whl>"
    fi
    check "TORCH_DELEGATE_ROOT=${TORCH_DELEGATE_ROOT}"

    local build_dir="$SCRIPT_DIR/build"
    rm -rf "$build_dir"
    mkdir -p "$build_dir"

    local torch_pkg_dir
    torch_pkg_dir="$(python3 -c 'import torch, os; print("TORCH_PKG_DIR_MARKER:" + os.path.dirname(torch.__file__))' 2>/dev/null \
        | grep '^TORCH_PKG_DIR_MARKER:' | cut -d: -f2-)" || true
    [[ -n "$torch_pkg_dir" ]] || fail "无法获取 TORCH_PKG_DIR"

    cmake -S "$SCRIPT_DIR" -B "$build_dir" \
        -DTorchDelegate_ROOT_DIR="${TORCH_DELEGATE_ROOT}" \
        -DCANN_ROOT="${ASCEND_HOME_PATH}" \
        -DTORCH_PKG_DIR="${torch_pkg_dir}" 2>&1 | tail -5 \
        || fail "cmake 配置失败"

    cmake --build "$build_dir" -j"$(nproc)" 2>&1 | tail -5 \
        || fail "cmake 构建失败"

    if [[ -f "$build_dir/epair_loader" ]]; then
        check "epair_loader 编译成功"
    else
        fail "epair_loader 编译失败"
    fi
}

# ---- [3/3] 运行 C++ EpairModelLoader ----
step3_run_cpp() {
    phase "运行 C++ EpairModelLoader"

    local build_dir="$SCRIPT_DIR/build"
    local epair_path="$SCRIPT_DIR/generate/simple_model.epair"

    if [[ ! -f "$build_dir/epair_loader" ]]; then
        skip "epair_loader 未构建"
        return 0
    fi

    if [[ ! -f "$epair_path" ]]; then
        skip "simple_model.epair 不存在, 请先运行步骤 1"
        return 0
    fi

    local torch_pkg_dir
    torch_pkg_dir="$(python3 -c 'import torch, os; print("TORCH_PKG_DIR_MARKER:" + os.path.dirname(torch.__file__))' 2>/dev/null \
        | grep '^TORCH_PKG_DIR_MARKER:' | cut -d: -f2-)" || true
    [[ -n "$torch_pkg_dir" ]] || fail "无法获取 TORCH_PKG_DIR"
    export LD_LIBRARY_PATH="${torch_pkg_dir}/lib:${torch_pkg_dir}.libs:${LD_LIBRARY_PATH:-}"

    echo "  执行: $build_dir/epair_loader $epair_path"
    echo ""
    "$build_dir/epair_loader" "$epair_path" 2>&1 || true
}

# ---- main ----
step1_export
step2_build_cpp
step3_run_cpp

echo ""
echo "========================================"
echo " 全部完成"
echo "========================================"
