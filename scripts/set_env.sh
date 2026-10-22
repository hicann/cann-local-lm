#!/bin/bash
# set_env.sh — converter+compile 统一 env (复用一套)
#   converter (①): P1 venv (CPU torch 2.9.0 + torchair, 无 torch_npu) + 真 CANN lib64,
#                   torchair GE 转换走真 libge_runner (本机有 NPU 设备, 无需 stubs)
#   compile   (②): cann_lm_compile + 真 CANN (aclgrphBuildModel), 无 python 依赖
#   ③ NPU 运行仍用 set_env_cann.sh --npu (系统 python + torch_npu + device)
#
# 不用 stubs、不装 torch_npu、不 pin 0715 (用本机已装 cann-toolkit, 通常 0828)。
# 前置: 先 source CANN set_env.sh (设 ASCEND_HOME_PATH)。
#
# 用法: source scripts/set_env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_common.sh" || return 1 2>/dev/null || exit 1   # 校验 ASCEND_HOME_PATH + 通用变量

# ── converter 用 ──
export TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME}"
# CONVERTER_PYTHON: 尊重外部预置 (whl 模式用户设自己 venv); 否则从 TORCH_DELEGATE_HOME/.venv 派生
export CONVERTER_PYTHON="${CONVERTER_PYTHON:-${TORCH_DELEGATE_HOME}/.venv/bin/python}"
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1   # GE 详细日志 (定位 not registered 等)

# ── compile 用 ──
export SOC_VERSION="${SOC_VERSION:-Ascend910_9362}"
export GE_FUSION_SWITCH_FILE="${GE_FUSION_SWITCH_FILE:-${PROJECT_ROOT}/cann_lm/converter/export/csrc/tools/fusion_autoub_off.cfg}"
export GE_EXTERNAL_WEIGHT="${GE_EXTERNAL_WEIGHT:-2}"

# ── 统一 LD: 真 CANN lib64 优先 (liberror_manager 取 host 版, 避免 devlib/aarch64 device 版干扰;
#   _torchair 闭包 libge_runner/libascendcl/libhccl/libruntime_common 均 NEEDED liberror_manager,
#   不需 devlib 的 libascend_hal.so — 后者只被 libacl_dvpp_mpi/libascend_dump/libascend_trace 等 DVPP/dump/ml 库引用).
#   cann set_env.sh 先设 devlib:lib64 (devlib 在前致 device 版 liberror_manager 被解析), 此处强制 lib64 到最前覆盖.
#   无 stubs, 幂等 (重复 lib64 无害, 最前者优先). ──
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# ── cann_lm 自定位 PYTHONPATH (反查 P1 venv site-packages; compile 不用 python, 无影响) ──
_CANN_LM_SITE=""
if [[ -x "${CONVERTER_PYTHON}" ]]; then
    _CANN_LM_SITE=$("${CONVERTER_PYTHON}" -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null)
    if [[ -n "${_CANN_LM_SITE}" && -d "${_CANN_LM_SITE}/cann_lm" ]]; then
        if [[ ":${PYTHONPATH:-}:" != *":${_CANN_LM_SITE}:"* ]]; then
            export PYTHONPATH="${_CANN_LM_SITE}:${PYTHONPATH:-}"
        fi
    else
        _CANN_LM_SITE=""
        echo "[set_env] warning: cann_lm 未装于 P1 venv; 跑 bash scripts/build.sh deps" >&2
    fi
fi

echo "[set_env] converter+compile 统一 env (真 CANN, 无 stubs)"
echo "  CANN        : ${ASCEND_HOME_PATH}"
echo "  venv python : ${CONVERTER_PYTHON}"
echo "  dopt python : ${DOPT_PYTHON} (subprocess 用, --quant-method gptq 触发)"
echo "  SOC=${SOC_VERSION}  GE_EXTERNAL_WEIGHT=${GE_EXTERNAL_WEIGHT}  FUSION=$(basename "${GE_FUSION_SWITCH_FILE}")"
echo "  cann_lm site: ${_CANN_LM_SITE:-<未发现, 跑 build.sh deps>}"
unset _CANN_LM_SITE
