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

# ② OM 编译用 0715 CANN: 0828 的 9e36de07b(2026-07-16) 给 ChunkGatedDeltaRule op_info 加了 fp32
# initial_state 选项, GE 在 910 (op_select_format null) 把 bf16 promote 成 fp32, 910 kernel 拒绝→编译失败;
# 即使 must_keep 强制 bf16, 0715→0828 kernel/编译器行为变化致输出乱码. 0715 op_info 只 bf16, 正常编译+出字(已验证).
# ③ NPU 运行仍用 0828 (系统 torch_npu 依赖; 0715 编的 OM 在 0828 runtime 跑正常).
if [[ "${1:-}" != "--npu" ]]; then
    export CANN_SET_ENV="${CANN_SET_ENV_COMPILE:-/home/developer/Ascend/cann-9.2.0-0715/cann/set_env.sh}"
    unset ASCEND_HOME_PATH 2>/dev/null || true   # 强制 _env_common.sh (重新) source 0715
fi

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
