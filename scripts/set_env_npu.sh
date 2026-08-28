#!/bin/bash
# set_env_npu.sh — ④ npu verify-runtime env (dopt GPTQ 标定 / NPU 出字 / OM 验证)
#   系统 Python + 真 torch_npu + 真 CANN, 无 stubs, 不产构件
#
# 用法: source scripts/set_env_npu.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_common.sh"

export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export DEVICE=npu

echo "[set_env] ④ npu verify-runtime (系统 Python + 真 torch_npu + 真 CANN)"
echo "  python : ${DOPT_PYTHON}"
echo "  lib64  : ${ASCEND_HOME_PATH}/lib64 (无 stubs)"
echo "  DEVICE : ${DEVICE}"
