#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# run_all.sh - 批量执行 examples 验证用例
#
# 前置条件：
#   1. ASCEND_HOME_PATH 已设置（source CANN set_env.sh）
#   2. 已构建并安装 torch_delegate whl（../build.sh 后执行 prepare_env.sh）
#      TORCH_DELEGATE_ROOT 由本脚本从已安装的 whl 自动推导，无需 source set_env.sh
#
# 用法：
#   bash run_all.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_FILE="${SCRIPT_DIR}/case_result.txt"
VENV_DIR="${SCRIPT_DIR}/venv_test"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
RESULTS=()

# ============================================
# 用例表：修改此处控制用例启用/禁用
#   格式: "目录名|脚本名|启用标志"
#   enabled  = 执行
#   disabled = 跳过（待适配）
#   ignored  = 忽略（非测试用例）
# ============================================
CASES=(
  "export_and_load_example|run.sh|enabled"
  "inplace_input|run.sh|enabled"
  "sym_int_case|run.sh|enabled"
  "sym_pack_optimization_case|run.sh|enabled"
  "assert_tensor_metadata_demo|run.sh|enabled"
  "accuracy_compare|run.sh|ignored"
  "fallback_inplace_optional_case|run.sh|enabled"
  "inplace_return_case|run.sh|enabled"
  "fallback_dynamic_optional_case|run.sh|enabled"
  "run_model_multi_instance|run.sh|enabled"
  "host_cpu_mixed_case|run.sh|enabled"
)

log() {
  echo "$1"
  echo "$1" >> "${RESULT_FILE}"
}

run_case() {
  local case_dir="$1"
  local case_script="$2"
  local case_path="${SCRIPT_DIR}/${case_dir}/${case_script}"

  echo ""
  echo "============================================="
  echo "  Run ${case_dir}"
  echo "============================================="
  echo ""

  if [[ ! -f "${case_path}" ]]; then
    log "  [FAIL] 脚本不存在: ${case_path}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    RESULTS+=("FAIL  ${case_dir}")
    return
  fi

  local start_time
  start_time=$(date +%s)

  if bash "${case_path}"; then
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ""
    echo "============================================="
    log "  Result: ${case_dir} PASS (${elapsed}s)"
    echo "============================================="
    PASS_COUNT=$((PASS_COUNT + 1))
    RESULTS+=("PASS  ${case_dir} (${elapsed}s)")
  else
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    echo ""
    echo "============================================="
    log "  Result: ${case_dir} FAIL (${elapsed}s)"
    echo "============================================="
    FAIL_COUNT=$((FAIL_COUNT + 1))
    RESULTS+=("FAIL  ${case_dir} (${elapsed}s)")
  fi
}

# ============================================
# 主流程
# ============================================

: > "${RESULT_FILE}"

log "========================================"
log " TorchDelegate Examples - run_all.sh"
log " $(date '+%Y-%m-%d %H:%M:%S')"
log "========================================"
log ""

TOTAL_STEPS=5
STEP=0

next_step() {
  STEP=$((STEP + 1))
  log "========================================"
  log " [Step ${STEP}/${TOTAL_STEPS}] $1"
  log "========================================"
}

# Step 1: 创建虚拟环境
next_step "准备虚拟环境"
if [[ -d "${VENV_DIR}" ]]; then
  log "  venv_test 已存在，跳过创建"
else
  log "  创建 venv_test 环境..."
  if ! bash "${SCRIPT_DIR}/prepare_env.sh" 2>&1 | tee -a "${RESULT_FILE}"; then
    log "  [ERROR] prepare_env.sh 执行失败"
    exit 1
  fi
  log "  venv_test 环境创建完成"
fi
log ""

# Step 2: 检查环境变量
next_step "检查环境变量"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  log "  [ERROR] ASCEND_HOME_PATH 未设置"
  log "  请先执行: source /path/to/Ascend/cann/set_env.sh"
  exit 1
fi
log "  ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
log ""

# Step 3: 激活虚拟环境
next_step "激活虚拟环境"
if [[ ! -d "${VENV_DIR}" ]]; then
  log "  [ERROR] venv_test 不存在: ${VENV_DIR}"
  log "  请检查 Step 1 是否成功创建虚拟环境"
  exit 1
fi

# shellcheck disable=SC1091
if ! source "${VENV_DIR}/bin/activate"; then
  log "  [ERROR] 激活虚拟环境失败: ${VENV_DIR}/bin/activate"
  exit 1
fi
log "  Python: $(python3 --version 2>&1), $(which python3)"

# torch_delegate 由 whl 安装到当前 venv，C++ 前缀即包目录
if [[ -z "${TORCH_DELEGATE_ROOT:-}" ]]; then
  TORCH_DELEGATE_ROOT="$(python3 -c 'import torch_delegate; print(torch_delegate.cmake_prefix())' 2>/dev/null || true)"
fi
if [[ -z "${TORCH_DELEGATE_ROOT}" ]]; then
  log "  [ERROR] 无法获取 TORCH_DELEGATE_ROOT"
  log "  请先构建并安装 whl: (cd .. && ./build.sh) && bash prepare_env.sh"
  deactivate 2>/dev/null || true
  exit 1
fi
export TORCH_DELEGATE_ROOT
log "  TORCH_DELEGATE_ROOT=${TORCH_DELEGATE_ROOT}"
log ""

# Step 4: 执行用例
next_step "执行用例"
log ""

for entry in "${CASES[@]}"; do
  IFS='|' read -r case_dir case_script case_status <<< "${entry}"

  case "${case_status}" in
    enabled)
      run_case "${case_dir}" "${case_script}"
      ;;
    disabled)
      log "  [SKIP] ${case_dir} (disabled, 待适配)"
      SKIP_COUNT=$((SKIP_COUNT + 1))
      RESULTS+=("SKIP  ${case_dir} (disabled)")
      ;;
    ignored)
      log "  [IGNORE] ${case_dir} (非测试用例)"
      ;;
    *)
      log "  [WARN] ${case_dir} 未知状态: ${case_status}"
      ;;
  esac
  log ""
done

# Step 5: 汇总结果
next_step "执行结果汇总"
log ""
for r in "${RESULTS[@]}"; do
  log "  ${r}"
done
log ""
log "  PASS: ${PASS_COUNT}  FAIL: ${FAIL_COUNT}  SKIP: ${SKIP_COUNT}"
log ""

if [[ ${FAIL_COUNT} -gt 0 ]]; then
  log "  [RESULT] 存在失败用例"
  deactivate 2>/dev/null || true
  exit 1
else
  # 确保只在所有用例执行结束出现 CI SUCCESS 标记 Run all examples success
  log "  [RESULT] 全部通过 Run all examples success"
  deactivate 2>/dev/null || true
  exit 0
fi
