#!/bin/bash
# set_env_cann.sh — ③ NPU verify-runtime env (系统 python + torch_npu + device)
#
# converter (①) + compile (②) 已合并到 set_env.sh (真 CANN, 无 stubs, 无 0715)。
# 本脚本仅 ③: 真 CANN + DEVICE=npu, 调用方用 $DOPT_PYTHON 跑 dopt GPTQ / OM 运行 / 出字 / 验证。
#
# 用法:
#   source scripts/set_env_cann.sh --npu    # ③ NPU
# 前置: 先 source CANN set_env.sh (设 ASCEND_HOME_PATH)。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/_env_common.sh" || return 1 2>/dev/null || exit 1   # 校验 ASCEND_HOME_PATH + 通用变量

# 真 CANN lib64 (无 stubs; ③ NPU 基底); 幂等 prepend
_CANN_LIB64="${ASCEND_HOME_PATH}/lib64"
if [[ ":${LD_LIBRARY_PATH:-}:" != *":${_CANN_LIB64}:"* ]]; then
    export LD_LIBRARY_PATH="${_CANN_LIB64}:${LD_LIBRARY_PATH:-}"
fi
unset _CANN_LIB64

export DEVICE=npu
echo "[set_env] ③ NPU env (真 CANN, 无 stubs)"
echo "  lib64 : ${ASCEND_HOME_PATH}/lib64"
echo "  DEVICE: ${DEVICE}  (用 \$DOPT_PYTHON=${DOPT_PYTHON} 跑 dopt/出字/OM 验证)"
