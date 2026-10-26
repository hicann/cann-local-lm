#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXAMPLES_DIR}/.." && pwd)"

STEP="${1:-all}"
DEVICE="${DEVICE:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python}"
COMPILE_DEVICE="${COMPILE_DEVICE:-npu:0}"
DEVICE_ID="${DEVICE_ID:-0}"

check_env_var() {
  local var_name="$1"
  local description="$2"
  local var_value

  eval "var_value=\${${var_name}:-}"

  if [[ -z "${var_value}" ]]; then
    echo "[Error] 环境变量 ${var_name} 未设置"
    echo "[Error] ${description}"
    echo "[Error] 请设置: export ${var_name}=/path/to/..."
    exit 1
  fi

  local expanded_path
  expanded_path=$(eval echo "${var_value}")

  if [[ ! -e "${expanded_path}" ]]; then
    echo "[Error] ${var_name} 指定的路径不存在: ${expanded_path}"
    exit 1
  fi

  echo "[Check] ${var_name}=${expanded_path} ✓"
}

check_cann_env() {
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[Check] ASCEND_HOME_PATH=${ASCEND_HOME_PATH} ✓"
    return 0
  fi

  for candidate in "/usr/local/Ascend/cann" "${HOME}/Ascend/cann"; do
    if [[ -f "${candidate}/set_env.sh" ]]; then
      echo "[Check] source ${candidate}/set_env.sh"
      set +u
      source "${candidate}/set_env.sh"
      set -u
      return 0
    fi
  done

  echo "[Error] CANN 环境未加载"
  echo "[Error] 请先执行: source /path/to/Ascend/cann/set_env.sh"
  exit 1
}

derive_torch_delegate_root() {
  if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
    TORCH_DELEGATE_ROOT="$(${PYTHON_BIN} -c 'import torch_delegate; print(torch_delegate.cmake_prefix())' 2>/dev/null || true)"
    export TORCH_DELEGATE_ROOT
  fi
  if [[ -n "${TORCH_DELEGATE_ROOT:-}" ]]; then
    echo "[Check] TORCH_DELEGATE_ROOT=${TORCH_DELEGATE_ROOT} ✓"
    echo "  注: 默认取自已安装的 torch_delegate whl；如需其他路径，设置 TORCH_DELEGATE_ROOT 环境变量"
  else
    echo "[Warning] TORCH_DELEGATE_ROOT 未设置，且当前 Python 未安装 torch_delegate"
    echo "[Warning] 请先 pip install <torch_delegate whl>"
    exit 1
  fi
}

run_step1() {
  echo "[step1] export_program: export ExportedProgram"
  cd "${ROOT_DIR}"
  TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}" \
    ${PYTHON_BIN} -m export_program \
    --output "${ROOT_DIR}/generate/sample.pt2" \
    --device "${DEVICE}"
}

run_step2() {
  echo "[step2] compile_to_air: compile AIR on ${COMPILE_DEVICE}"
  cd "${ROOT_DIR}"

  echo ""
  echo "=============================================="
  echo "环境变量检查"
  echo "=============================================="
  check_cann_env
  echo ""

  ${PYTHON_BIN} -m compile_to_air \
    --ep-path "${ROOT_DIR}/generate/sample.pt2" \
    --air-path "${ROOT_DIR}/generate/sample.air" \
    --compile-device "${COMPILE_DEVICE}"
}

run_compare() {
  echo "[compare] accuracy: torch.compile(npu) vs C++ epair runner"
  cd "${ROOT_DIR}"

  echo ""
  echo "=============================================="
  echo "环境变量检查"
  echo "=============================================="

  check_cann_env
  derive_torch_delegate_root

  # 确保 libtorch_npu 已构建（公共依赖）
  source "${EXAMPLES_DIR}/common/ensure_libtorch_npu.sh"
  ensure_libtorch_npu || exit 1

  if [[ -z "${TORCH_PKG_DIR:-}" ]]; then
    TORCH_PKG_DIR=$(${PYTHON_BIN} -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null || echo "")
    if [[ -z "${TORCH_PKG_DIR}" ]]; then
      echo "[Error] 无法自动检测 TORCH_PKG_DIR"
      echo "[Error] 请设置: export TORCH_PKG_DIR=/path/to/site-packages/torch"
      exit 1
    fi
    echo "[Check] TORCH_PKG_DIR=${TORCH_PKG_DIR} (自动检测) ✓"
  else
    check_env_var "TORCH_PKG_DIR" "Python torch 包目录"
  fi
  TORCH_LIB_DIR="${TORCH_PKG_DIR}/lib"

  echo ""

  STUB_KERNEL_DIR="${PROJECT_ROOT}/tests/cpp/stub_kernel"
  STUB_KERNEL_SO="${STUB_KERNEL_DIR}/build/libstub_kernel.so"

  # Auto-detect stub_kernel from local build
  if [[ ! -f "${STUB_KERNEL_SO}" ]]; then
    if [[ -f "${STUB_KERNEL_SO}" ]]; then
      echo "[Check] stub_kernel.so: ${STUB_KERNEL_SO} (auto-detected) ✓"
    else
      echo "[Error] stub_kernel.so not found"
      echo "[Error] 请先执行: bash run.sh build-deps"
      exit 1
    fi
  else
    echo "[Check] stub_kernel.so: ${STUB_KERNEL_SO} ✓"
  fi

  echo "[compare] Building epair_runner..."
  cd "${ROOT_DIR}/epair_runner"
  rm -rf build
  cmake -B build \
    -DTorchDelegate_ROOT="${TORCH_DELEGATE_ROOT}" \
    -DCANN_ROOT="${ASCEND_HOME_PATH}" \
    -DTORCH_PKG_DIR="${TORCH_PKG_DIR}"
  cmake --build build -j$(nproc)
  cd "${ROOT_DIR}"

  echo "[compare] Building custom_op..."
  cd "${ROOT_DIR}/custom_op"
  rm -rf build
  cmake -B build \
    -DTORCH_PKG_DIR="${TORCH_PKG_DIR}" \
    -DTORCH_NPU_DIR="${TORCH_NPU_DIR}" \
    -DASCEND_HOME_PATH="${ASCEND_HOME_PATH}"
  cmake --build build -j$(nproc)
  cd "${ROOT_DIR}"

  # xllm_ops 算子包安装路径（由 build_accuracy_deps.sh 安装到 CANN）
  local xllm_ops_lib="${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math/op_api/lib"

  export LD_LIBRARY_PATH="${ROOT_DIR}/custom_op/build:${TORCH_PKG_DIR}.libs:${TORCH_LIB_DIR}:${TORCH_DELEGATE_ROOT}/lib:${xllm_ops_lib}:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
  export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=1
  export TORCH_PKG_DIR="${TORCH_PKG_DIR}"
  export TORCH_NPU_DIR="${TORCH_NPU_DIR}"

  echo ""
  echo "=============================================="
  echo "环境变量（执行阶段）"
  echo "=============================================="
  echo "  CANN 环境:"
  echo "    ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
  echo "  TorchDelegate:"
  echo "    TORCH_DELEGATE_ROOT=${TORCH_DELEGATE_ROOT}"
  echo "  Torch/torch_npu:"
  echo "    TORCH_PKG_DIR=${TORCH_PKG_DIR}"
  echo "    TORCH_LIB_DIR=${TORCH_LIB_DIR}"
  echo "    TORCH_NPU_DIR=${TORCH_NPU_DIR}"
  echo "  其他:"
  echo "    STUB_KERNEL_SO=${STUB_KERNEL_SO}"
  echo "    TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=${TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN}"
  echo ""

  # 执行精度对比，执行完毕后无论成功与否都卸载 xllm_ops 算子包
  local compare_rc=0
  ${PYTHON_BIN} -m accuracy_compare \
    --ep-path "${ROOT_DIR}/generate/sample.rewritten.pt2" \
    --epair-path "${ROOT_DIR}/generate/rewritten.epair" \
    --epair-runner "${ROOT_DIR}/epair_runner/build/epair_runner" \
    --output-json "${ROOT_DIR}/generate/accuracy_result.json" \
    --device "${COMPILE_DEVICE}" \
    --seed 0 || compare_rc=$?

  echo ""
  echo "=============================================="
  echo "卸载 xllm_ops 算子包"
  echo "=============================================="
  uninstall_xllm_ops

  if [[ ${compare_rc} -ne 0 ]]; then
    echo "[Error] accuracy_compare 执行失败 (rc=${compare_rc})"
    exit ${compare_rc}
  fi
}

uninstall_xllm_ops() {
  local vendor_dir="${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math"
  local uninstall_script="${vendor_dir}/scripts/uninstall.sh"
  if [[ -f "${uninstall_script}" ]]; then
    echo "[Cleanup] 执行 xllm_ops 卸载脚本..."
    bash "${uninstall_script}" 2>&1 || true
    echo "[OK] xllm_ops 算子包已卸载"
  else
    if [[ -d "${vendor_dir}" ]]; then
      echo "[Cleanup] 未找到卸载脚本，手动删除 ${vendor_dir}"
      chmod -R +w "${vendor_dir}" 2>/dev/null || true
      rm -rf "${vendor_dir}"
      echo "[OK] xllm_ops 算子包已删除"
    else
      echo "[Info] xllm_ops 算子包不存在，无需卸载"
    fi
  fi

  # 清理 config.ini 中的 load_priority 条目
  local config_file="${ASCEND_HOME_PATH}/opp/vendors/config.ini"
  if [[ -f "${config_file}" ]]; then
    sed -i '/custom_xllm_math/d' "${config_file}" 2>/dev/null || true
    # 若 config.ini 为空则删除
    if [[ ! -s "${config_file}" ]]; then
      rm -f "${config_file}"
    fi
  fi
}

run_build_deps() {
  echo "[build_deps] 构建专有依赖 (xllm_ops, stub_kernel)..."
  echo "[build_deps] libtorch_npu 由公共脚本自动处理"
  cd "${ROOT_DIR}"
  bash "${ROOT_DIR}/build_accuracy_deps.sh"
}

# 自动触发 deps 编译（已存在则跳过）
run_build_deps

case "${STEP}" in
  step1)
    run_step1
    ;;
  step2)
    run_step2
    ;;
  compare)
    run_compare
    ;;
  build-deps)
    ;;
  all)
    run_step1
    run_step2
    run_compare
    ;;
  *)
    echo "Usage: bash run.sh [step1|step2|compare|build-deps|all]"
    echo ""
    echo "  all (default)  - step1 + step2 + compare（自动构建依赖，执行后卸载 xllm_ops）"
    echo "  build-deps     - 仅构建专有依赖 (xllm_ops, stub_kernel)"
    echo "  step1          - 导出 ExportedProgram"
    echo "  step2          - 编译 AIR"
    echo "  compare        - 精度比对（执行后卸载 xllm_ops）"
    exit 1
    ;;
esac
