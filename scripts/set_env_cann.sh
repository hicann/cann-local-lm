#!/bin/bash
# set_env_cann.sh — real-CANN env (② OM编译 / ③ NPU verify-runtime)
#
# ②③ 同 LD 基底 (真 CANN lib64 在前, 无 stubs); 差异:
#   不带 --npu (②): OM 编译, 真 CANN, 无 python/torch; 额外带编译 flag (SOC + fusion cfg + 权值分离)
#   --npu        (③): 真 CANN + DEVICE=npu (调用方用 $DOPT_PYTHON 跑 dopt GPTQ / OM 运行 / 出字 / 验证)
#
# 用法:
#   source scripts/set_env_cann.sh          # ② OM 编译
#   source scripts/set_env_cann.sh --npu    # ③ NPU
#
# 注: source 不带显式参时会继承父 shell 的 $1; 仅当 $1=="--npu" 才走 ③, 其余皆 ②.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_common.sh"

# 真 CANN lib64 在前 (无 stubs; ② 编译 / ③ NPU 共用基底)
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

if [[ "${1:-}" == "--npu" ]]; then
    # ③ NPU: 只加 device (dopt/出字/OM 验证)
    export DEVICE=npu
    echo "[set_env] real-CANN env (③ NPU)"
    echo "  lib64 : ${ASCEND_HOME_PATH}/lib64 (无 stubs)"
    echo "  DEVICE: ${DEVICE}  (用 \$DOPT_PYTHON=${DOPT_PYTHON} 跑 dopt/出字)"
else
    # ② OM 编译: 编译 flag (SOC + fusion cfg 关 AUB + 权值分离; ③ 不需要)
    export SOC_VERSION="${SOC_VERSION:-Ascend910_9362}"
    export GE_FUSION_SWITCH_FILE="${GE_FUSION_SWITCH_FILE:-${PROJECT_ROOT}/cann_lm/converter/export/csrc/tools/fusion_autoub_off.cfg}"
    export GE_EXTERNAL_WEIGHT="${GE_EXTERNAL_WEIGHT:-2}"
    echo "[set_env] real-CANN env (② OM 编译)"
    echo "  lib64 : ${ASCEND_HOME_PATH}/lib64 (无 stubs)"
    echo "  SOC=${SOC_VERSION}  GE_EXTERNAL_WEIGHT=${GE_EXTERNAL_WEIGHT}  FUSION=$(basename "${GE_FUSION_SWITCH_FILE}")"
fi
