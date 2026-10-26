#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# run.sh - run_model_multi_instance 一键式示例脚本
#
# 验证 td::EpairModelLoader::RunModel 的多进程并发执行（一进程一实例）：
#   1. 导出“深而宽”的 matmul 模型为 .epair 及输入数据（多层 + 每层多分支）
#   2. 构建 C++ 示例
#   3. 拉起 INSTANCES 个 worker 进程（每进程独立 GeSession/ACL 上下文/Device 流），
#      worker 内部为生产者线程 + 消费线程，并行调用各自的 RunModel
#   4. 校验：全部任务成功 + checksum 一致 + 并发门禁
#      - host 侧并发下发：host_max_inflight >= 2
#      - device 侧多流并行：device_parallelism >= INSTANCES * MIN_PARALLEL_RATIO
#
# 前置条件：
#   1. source <cann_path>/cann/set_env.sh
#   2. python3 环境已激活，torch_npu/torch_delegate(whl) 已安装
#      TORCH_DELEGATE_ROOT 会自动从已安装的 whl 推导，无需 source set_env.sh
#
# 用法：
#   bash run.sh
#
# 可通过环境变量覆盖参数：
#   INSTANCES=8 TASKS=48 DIM=1024 DEPTH=24 BRANCHES=8 bash run.sh
#   MIN_PARALLEL_RATIO=0.5 bash run.sh         # 并发门禁比例
#   TD_MULTI_TIMEOUT_S=900 bash run.sh         # 栅栏超时（秒）

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE=0
TOTAL=5
INSTANCES="${INSTANCES:-8}"
TASKS="${TASKS:-48}"
DIM="${DIM:-1024}"
DEPTH="${DEPTH:-24}"
BRANCHES="${BRANCHES:-8}"
MIN_PARALLEL_RATIO="${MIN_PARALLEL_RATIO:-0.5}"

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

if ! python3 -c "import torch, torch_npu; assert torch.npu.is_available()" 2>/dev/null; then
    fail "NPU 不可用（请确认已挂载设备并安装驱动，torch.npu.is_available() 为 True）"
fi
check "NPU available"

# ============================================
# Step 2: 导出深而宽的 matmul epair + 输入数据
# ============================================
phase "导出 matmul epair (dim=${DIM}, depth=${DEPTH}, branches=${BRANCHES})"

python3 "${ROOT_DIR}/export_matmul_epair.py" \
    --dim "${DIM}" --depth "${DEPTH}" --branches "${BRANCHES}" || fail "模型导出失败"

if [[ -f "${ROOT_DIR}/matmul_model.epair" ]]; then
    check "${ROOT_DIR}/matmul_model.epair"
else
    fail "matmul_model.epair 未生成"
fi
if [[ -f "${ROOT_DIR}/input_0.bin" ]]; then
    check "${ROOT_DIR}/input_0.bin"
else
    fail "input_0.bin 未生成"
fi

# ============================================
# Step 3: 构建 C++ 示例
# ============================================
phase "构建 C++ 示例"

cd "${ROOT_DIR}"
rm -rf build
_TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || fail "无法获取 TORCH_PKG_DIR"
cmake -S . -B build \
    -DTorchDelegate_ROOT_DIR="${TORCH_DELEGATE_ROOT}" \
    -DCANN_ROOT="${ASCEND_HOME_PATH}" \
    -DTORCH_PKG_DIR="${_TORCH_PKG_DIR}" > /dev/null 2>&1 || fail "cmake 配置失败"
cmake --build build -j"$(nproc)" > /dev/null 2>&1 || fail "cmake 构建失败"

if [[ -x "${ROOT_DIR}/build/run_model_multi_instance" ]]; then
    check "build/run_model_multi_instance"
else
    fail "编译失败"
fi

# ============================================
# Step 4: 多进程并发运行
# ============================================
phase "多进程并发运行 (loaders=${INSTANCES}, tasks=${TASKS})"

TORCH_PKG_DIR="$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)" || fail "无法获取 torch 路径"
export LD_LIBRARY_PATH="${TORCH_PKG_DIR}.libs:${TORCH_PKG_DIR}/lib:${LD_LIBRARY_PATH:-}"

RUN_LOG="${ROOT_DIR}/run.log"
set +e
"${ROOT_DIR}/build/run_model_multi_instance" "${ROOT_DIR}/matmul_model.epair" \
    "${INSTANCES}" "${TASKS}" "${DIM}" 2>&1 | tee "${RUN_LOG}"
run_rc=${PIPESTATUS[0]}
set -e

if [[ ${run_rc} -ne 0 ]]; then
    fail "示例运行失败 (exit=${run_rc})"
fi
check "示例执行完成"

# ============================================
# Step 5: 校验结果（正确性 + 并发门禁）
# ============================================
phase "校验结果"

if ! grep -q "success / fail  : ${TASKS} / 0" "${RUN_LOG}"; then
    fail "未看到全部任务成功（期望 success=${TASKS}, fail=0）"
fi
check "全部 ${TASKS} 个任务成功"

if ! grep -q "checksum mismatch: 0" "${RUN_LOG}"; then
    fail "输出 checksum 不一致"
fi
check "输出 checksum 一致"

# host 侧并发下发：RunModel 同时进入的线程数 >= 2
inflight="$(grep -o 'host_max_inflight : [0-9]\+' "${RUN_LOG}" | tail -1 | awk '{print $3}')"
if [[ -z "${inflight}" ]]; then
    fail "未解析到 host_max_inflight 指标"
fi
check "host 侧并发进入线程数：host_max_inflight=${inflight}"

# device 侧多流并行：device_parallelism >= INSTANCES * MIN_PARALLEL_RATIO
dev_par="$(grep -o 'device_parallelism : [0-9.]\+' "${RUN_LOG}" | tail -1 | awk '{print $3}')"
threshold="$(awk -v n="${INSTANCES}" -v r="${MIN_PARALLEL_RATIO}" 'BEGIN{printf "%.2f", n*r}')"
if [[ -z "${dev_par}" ]]; then
    fail "未解析到 device_parallelism 指标"
fi
if awk -v v="${dev_par}" -v t="${threshold}" 'BEGIN{exit !(v < t)}'; then
    echo "  device_parallelism=${dev_par} < ${threshold}=${INSTANCES}*${MIN_PARALLEL_RATIO}"
    echo "  [原因] 并发不足。用 msprof 抓 timeline 检查 device 是否串行执行。"
    fail "并发门禁未通过"
fi
check "device 侧多流并行：device_parallelism=${dev_par} >= ${threshold}"
if [[ "${inflight}" -lt 2 ]]; then
    fail "host 侧未并发下发算子（host_max_inflight=${inflight} < 2）"
fi
check "host 侧并发下发：${inflight} 个进程同时进入 RunModel"

echo ""
grep -E "mode|success / fail|checksum mismatch|wall|t_solo|avg concurrency|host_max_inflight|device_parallelism" "${RUN_LOG}" || true

echo ""
echo "========================================"
echo " 验证通过"
echo "========================================"
echo ""
echo "可选：用 msprof 抓 timeline 验证 device 多流（各进程算子落在不同 stream 且时间"
echo "区间重叠）："
echo "  msprof --output=./prof_multi_instance --application=\"${ROOT_DIR}/build/run_model_multi_instance ${ROOT_DIR}/matmul_model.epair ${INSTANCES} ${TASKS} ${DIM}\""
echo "  之后查看 ${ROOT_DIR}/prof_multi_instance/ 下的 timeline（mindstudio 或 msprof 报表）"
