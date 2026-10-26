#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# run.sh - fallback_dynamic_optional_case 一键式验证脚本
#
# 前置条件：
#   1. source <cann_path>/cann/set_env.sh
#   2. venv 已激活（由 run_all.sh 或手动激活）
#   3. TORCH_NPU_DIR 可选（未设置时自动从 torch_npu 包检测）
#
# 用法：
#   bash run.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXAMPLES_DIR}/.." && pwd)"
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
  echo "  [✓] $1"
}

fail() {
  echo "  [✗] $1"
  exit 1
}

phase "检查环境"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  fail "ASCEND_HOME_PATH 未设置，请先 source CANN 环境"
fi
check "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

if ! command -v python3 &> /dev/null; then
  fail "python3 未找到"
fi
check "python3 $(python3 --version 2>&1 | awk '{print $2}')"

if ! command -v cmake &> /dev/null; then
  fail "cmake 未找到"
fi
check "cmake $(cmake --version | head -1 | awk '{print $3}')"

phase "准备 libtorch_npu"

source "${EXAMPLES_DIR}/common/ensure_libtorch_npu.sh"
ensure_libtorch_npu || fail "libtorch_npu 准备失败"

phase "导出模型"
cd "${ROOT_DIR}"
python3 export_model.py

EPAIRS=""
for model_name in model_addn model_addmm_no_bias model_addmm_with_bias; do
  if [[ -f "${ROOT_DIR}/output/${model_name}.epair" ]]; then
    check "${model_name}.epair 已生成"
    EPAIRS="$EPAIRS output/${model_name}.epair"
  fi
done

phase "构建 C++ Runner + libcustom_ops"
cd "${ROOT_DIR}"
rm -rf build
mkdir -p build
cd build

TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || fail "无法获取 torch 路径"

if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
  TORCH_DELEGATE_ROOT="$(python3 -c 'import torch_delegate; print(torch_delegate.cmake_prefix())' 2>/dev/null || true)"
  export TORCH_DELEGATE_ROOT
fi

CMAKE_DELEGATE_ARGS=""
if [[ -n "${TORCH_DELEGATE_ROOT:-}" ]] && [[ -d "${TORCH_DELEGATE_ROOT}/lib" ]]; then
  check "TORCH_DELEGATE_ROOT=${TORCH_DELEGATE_ROOT} (已安装包)"
  CMAKE_DELEGATE_ARGS="-DTorchDelegate_ROOT_DIR=${TORCH_DELEGATE_ROOT}"
  DELEGATE_LIB_DIR="${TORCH_DELEGATE_ROOT}/lib"
elif [[ -n "${TORCH_DELEGATE_BUILD_DIR:-}" ]] && [[ -d "${TORCH_DELEGATE_BUILD_DIR}/csrc" ]]; then
  check "TORCH_DELEGATE_BUILD_DIR=${TORCH_DELEGATE_BUILD_DIR} (源码构建)"
  CMAKE_DELEGATE_ARGS="-DTORCH_DELEGATE_BUILD_DIR=${TORCH_DELEGATE_BUILD_DIR}"
  DELEGATE_LIB_DIR="${TORCH_DELEGATE_BUILD_DIR}/csrc"
elif [[ -d "${PROJECT_ROOT}/build/release/csrc" ]]; then
  check "TORCH_DELEGATE_BUILD_DIR=${PROJECT_ROOT}/build/release (源码构建，自动检测)"
  CMAKE_DELEGATE_ARGS="-DTORCH_DELEGATE_BUILD_DIR=${PROJECT_ROOT}/build/release"
  DELEGATE_LIB_DIR="${PROJECT_ROOT}/build/release/csrc"
else
  fail "无法找到 torch_delegate 库。请设置 TORCH_DELEGATE_ROOT（已安装包）或 TORCH_DELEGATE_BUILD_DIR（源码构建）"
fi

cmake .. \
  -DCANN_ROOT="${ASCEND_HOME_PATH}" \
  -DCMAKE_BUILD_TYPE=Debug \
  ${CMAKE_DELEGATE_ARGS} \
  -DTORCH_PKG_DIR="${TORCH_PKG_DIR}" \
  -DTORCH_NPU_DIR="${TORCH_NPU_DIR}"
cmake --build . -j"$(nproc)"

if [[ ! -f "fallback_runner" ]]; then
  fail "fallback_runner 编译失败"
fi
check "fallback_runner 已编译"

if [[ ! -f "libcustom_ops.so" ]]; then
  fail "libcustom_ops.so 编译失败"
fi
check "libcustom_ops.so 已编译"

phase "执行 Runner"
cd "${ROOT_DIR}"

export LD_LIBRARY_PATH="${TORCH_PKG_DIR}/lib:${TORCH_PKG_DIR}.libs:${TORCH_NPU_DIR}/lib:${DELEGATE_LIB_DIR}:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_PKG_DIR="${TORCH_PKG_DIR}"
export TORCH_NPU_DIR="${TORCH_NPU_DIR}"
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
export TORCH_DELEGATE_OUTPUT_DIR="${ROOT_DIR}/output_dir"
export TORCH_DELEGATE_KEEP_ARTIFACTS=1
export TORCH_DELEGATE_LOG_LEVEL=INFO
export TASK_QUEUE_ENABLE=0

rm -rf output_dir
mkdir -p output_dir

echo "执行模型:${EPAIRS}"
./build/fallback_runner --custom-ops build/libcustom_ops.so ${EPAIRS}

echo ""
echo "检查产物:"
for model_dir in output_dir/*/; do
  if [[ -d "$model_dir" ]]; then
    echo "  模型: $(basename "$model_dir")"
    for f in custom_op.cpp CMakeLists.txt libcust_opapi.so; do
      if [[ -f "${model_dir}${f}" ]]; then
        check "$f ($(stat -c%s "${model_dir}${f}") bytes)"
      else
        echo "  [✗] $f 未生成"
      fi
    done
  fi
done

phase "精度对比"
cd "${ROOT_DIR}"
python3 compare_accuracy.py
check "精度对比通过"

echo ""
echo "========================================"
echo " 验证通过"
echo "========================================"
