#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# run.sh - sym_pack_optimization_case 一键式验证脚本
#
# 验证两条路径的 Pack 生成与优化:
#   1. torchair torch.compile(dynamic=True): optimize_sym_pack 消除冗余 Pack
#   2. torch.export + epair: 导出含 Pack 的 epair, C++ EpairModelLoader 加载推理
#
# 前置条件：
#   1. source <cann_path>/cann/set_env.sh
#   2. 当前 Python 环境已安装 torch_delegate whl
#      （TORCH_DELEGATE_ROOT 会自动从已安装包推导，无需 source set_env.sh）
#
# 用法：
#   bash run.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE=0
TOTAL=6

phase() {
    PHASE=$((PHASE + 1))
    echo ""
    echo "========================================"
    echo " [${PHASE}/${TOTAL}] $1"
    echo "========================================"
}

check() {
    echo "  [OK] $1"
}

fail() {
    echo "  [FAIL] $1"
    exit 1
}

# ============================================
# Step 1: 环境检查
# ============================================
phase "检查环境"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    fail "ASCEND_HOME_PATH 未设置，请先 source CANN 环境"
fi
check "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
    TORCH_DELEGATE_ROOT="$(python3 -c 'import torch_delegate; print(torch_delegate.cmake_prefix())' 2>/dev/null || true)"
    export TORCH_DELEGATE_ROOT
fi
if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
    fail "TORCH_DELEGATE_ROOT 未设置，且当前 Python 未安装 torch_delegate；请先 pip install <torch_delegate whl>"
fi
check "TORCH_DELEGATE_ROOT=${TORCH_DELEGATE_ROOT}"

if ! command -v python3 &> /dev/null; then
    fail "python3 未找到"
fi
check "python3 $(python3 --version 2>&1 | awk '{print $2}')"

if ! command -v cmake &> /dev/null; then
    fail "cmake 未找到"
fi
check "cmake $(cmake --version | head -1 | awk '{print $3}')"

# ============================================
# Step 2: torchair torch.compile 路径 Pack 优化
# ============================================
phase "torchair torch.compile: optimize_sym_pack"

python3 "${ROOT_DIR}/torchair_pack_optimization.py"
check "torchair Pack 优化验证完成"

# 展示 pbtxt dump 结果
dump_dir="${ROOT_DIR}/graph_dump_output"
if ls "${dump_dir}"/*.pbtxt &>/dev/null 2>&1; then
    latest_orig=$(ls -t "${dump_dir}"/dynamo_original_*.pbtxt 2>/dev/null | head -1)
    latest_opt=$(ls -t "${dump_dir}"/dynamo_optimized_*.pbtxt 2>/dev/null | head -1)
    for label_entry in "original:${latest_orig}" "optimized:${latest_opt}"; do
        label="${label_entry%%:*}"
        f="${label_entry#*:}"
        [[ -z "$f" || ! -f "$f" ]] && continue
        pack_cnt=$(grep -c 'op: "Pack"' "$f" 2>/dev/null || echo 0)
        data_cnt=$(grep -c 'op: "Data"' "$f" 2>/dev/null || echo 0)
        printf "    [%s] Pack=%s  Data=%s\n" "$label" "$pack_cnt" "$data_cnt"
    done
fi

# ============================================
# Step 3: torch.export + epair 导出
# ============================================
phase "torch.export + epair: 导出 + Pack 分析"

python3 "${ROOT_DIR}/export_pack_model.py"
EPAIR_PATH="${ROOT_DIR}/generated/model.epair"
if [[ -f "${EPAIR_PATH}" ]]; then
    check "model.epair 已生成"
else
    fail "model.epair 未生成"
fi

# ============================================
# Step 4: 构建 C++ 示例
# ============================================
phase "构建 C++ EpairModelLoader"

cd "${ROOT_DIR}"
rm -rf build
_TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || fail "无法获取 TORCH_PKG_DIR"
cmake -S . -B build \
    -DTorchDelegate_ROOT_DIR="${TORCH_DELEGATE_ROOT}" \
    -DCANN_ROOT="${ASCEND_HOME_PATH}" \
    -DTORCH_PKG_DIR="${_TORCH_PKG_DIR}" > /dev/null 2>&1 || fail "cmake 配置失败"
cmake --build build -j$(nproc) > /dev/null 2>&1 || fail "cmake 构建失败"
if [[ -f "${ROOT_DIR}/build/sym_pack_optimization_case" ]]; then
    check "build/sym_pack_optimization_case"
else
    fail "编译失败"
fi

# ============================================
# Step 5: C++ 推理 + 精度比对
# ============================================
phase "C++ EpairModelLoader 推理 + 精度比对"

cd "${ROOT_DIR}"
TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || fail "无法获取 torch 路径"
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}.libs:${TORCH_PKG_DIR}/lib:${LD_LIBRARY_PATH:-}"

# 配置 GE Dump: 导出编译后的图, 校验 SymPack 优化已消除所有 Pack 算子
DUMP_DIR="${ROOT_DIR}/ge_graph_dump"
rm -rf "${DUMP_DIR}"
mkdir -p "${DUMP_DIR}"
export DUMP_GE_GRAPH=2
export DUMP_GRAPH_LEVEL=3
export DUMP_GRAPH_PATH="${DUMP_DIR}"

if ! "${ROOT_DIR}/build/sym_pack_optimization_case" "${EPAIR_PATH}" 2>&1; then
    fail "C++ 推理失败"
fi
check "C++ 推理完成"

unset DUMP_GE_GRAPH DUMP_GRAPH_LEVEL DUMP_GRAPH_PATH

# ============================================
# Step 6: 校验 dump 图中不存在 Pack 算子
# ============================================
phase "校验 GE Dump 图: 不存在 Pack 算子"

if [[ ! -d "${DUMP_DIR}" ]] || [[ -z "$(ls -A "${DUMP_DIR}" 2>/dev/null)" ]]; then
    fail "GE Dump 目录为空: ${DUMP_DIR}"
fi

pack_count=$(grep -rl 'type: "Pack"' "${DUMP_DIR}" --include='ge_proto*.txt' 2>/dev/null | wc -l || true)

echo "  搜索模式: 'type: \"Pack\"' in ge_proto*.txt (递归)"
echo "  Pack 算子文件数: ${pack_count}"

if [[ "$pack_count" -eq 0 ]]; then
    check "GE Dump 图中不存在 Pack 算子, SymPack 优化已生效"
else
    fail "GE Dump 图中仍存在 Pack 算子"
fi

echo ""
echo "========================================"
echo " 验证通过"
echo "========================================"
