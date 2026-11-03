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
# 用法: source scripts/cann_lm_compile/set_env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_common.sh" || return 1 2>/dev/null || exit 1   # 校验 ASCEND_HOME_PATH + 通用变量

# ── converter 用 ──
export TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME}"
# CONVERTER_PYTHON: 尊重外部预置 (whl 模式用户设自己 venv); 否则从 TORCH_DELEGATE_HOME/.venv 派生
export CONVERTER_PYTHON="${CONVERTER_PYTHON:-${TORCH_DELEGATE_HOME}/.venv/bin/python}"
export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1   # GE 详细日志 (定位 not registered 等)

# ── compile 用 ──
# 环境变量说明 (与发布包 set_env.sh 语义一致, 区别仅在插件目录取自源码树):
#   SOC_VERSION               目标 SoC (ge.socVersion)。Kirin 系 → mobile 编译链
#                             (.omc + MobileModelHelper); 其余 → 通用 .om 路径。
#   GE_FUSION_SWITCH_FILE     GE 图融合开关配置 (默认仓内 fusion_autoub_off.cfg,
#                             关 auto-UB 融合, 避免 mobile 路径出无 kernel 的融合子图)。
#   GE_EXTERNAL_WEIGHT        2 = 权值外置 (<out>.omc + weight/*.bin); 0 = 内嵌。
#   GE_MOBILE_WEIGHT_SHARE_OPS
#                             mobile 共享权重算子白名单 (GE [Mobile] 路径读取)。
#   ASCEND_CUSTOM_OPP_PATH    GE python 自定义算子插件目录 (op_cpu: Bitcast/TransData HostCpu)。
#   ASCEND_GE_PY_PASS_PATH    GE python pass 插件目录 (op_pass: QbmX2TransDataPass)。
#   后四项仅 SOC_VERSION 为 Kirin 系时自动设置。
export SOC_VERSION="${SOC_VERSION:-Ascend910_9362}"
export GE_FUSION_SWITCH_FILE="${GE_FUSION_SWITCH_FILE:-${PROJECT_ROOT}/cann_local_lm/converter/export/csrc/tools/fusion_autoub_off.cfg}"
export GE_EXTERNAL_WEIGHT="${GE_EXTERNAL_WEIGHT:-2}"

# Kirin 系 SOC → mobile 编译链 (与 compile_main.cc / ge_converter.py 的判别口径一致)
# 开发态插件目录 = 源码树 cann_local_lm/converter/export/{op_cpu,op_pass}
case "$SOC_VERSION" in
    Kirin*|kirin*)
        export GE_MOBILE_WEIGHT_SHARE_OPS="${GE_MOBILE_WEIGHT_SHARE_OPS:-MatMulV3,QuantBatchMatmulV4,RmsNorm}"
        _DEV_OP_CPU="${PROJECT_ROOT}/cann_local_lm/converter/export/op_cpu"
        _DEV_OP_PASS="${PROJECT_ROOT}/cann_local_lm/converter/export/op_pass"
        # 幂等: 重复 source 不重复追加
        [[ ":${ASCEND_CUSTOM_OPP_PATH:-}:" == *":${_DEV_OP_CPU}:"* ]] \
            || export ASCEND_CUSTOM_OPP_PATH="${_DEV_OP_CPU}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
        [[ ":${ASCEND_GE_PY_PASS_PATH:-}:" == *":${_DEV_OP_PASS}:"* ]] \
            || export ASCEND_GE_PY_PASS_PATH="${_DEV_OP_PASS}${ASCEND_GE_PY_PASS_PATH:+:${ASCEND_GE_PY_PASS_PATH}}"
        [[ ":${LD_LIBRARY_PATH:-}:" == *":${_DEV_OP_CPU}:"* ]] \
            || export LD_LIBRARY_PATH="${_DEV_OP_CPU}:${LD_LIBRARY_PATH:-}"
        _MOBILE=1
        ;;
    *)
        _MOBILE=0
        ;;
esac

# ── 统一 LD: 真 CANN lib64 优先 (liberror_manager 取 host 版, 避免 devlib/aarch64 device 版干扰;
#   _torchair 闭包 libge_runner/libascendcl/libhccl/libruntime_common 均 NEEDED liberror_manager,
#   不需 devlib 的 libascend_hal.so — 后者只被 libacl_dvpp_mpi/libascend_dump/libascend_trace 等 DVPP/dump/ml 库引用).
#   cann set_env.sh 先设 devlib:lib64 (devlib 在前致 device 版 liberror_manager 被解析), 此处强制 lib64 到最前覆盖.
#   无 stubs, 幂等 (重复 lib64 无害, 最前者优先). ──
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"

# ── cann_local_lm 自定位 PYTHONPATH (反查 P1 venv site-packages; compile 不用 python, 无影响) ──
_CANN_LM_SITE=""
if [[ -x "${CONVERTER_PYTHON}" ]]; then
    _CANN_LM_SITE=$("${CONVERTER_PYTHON}" -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null)
    if [[ -n "${_CANN_LM_SITE}" && -d "${_CANN_LM_SITE}/cann_local_lm" ]]; then
        if [[ ":${PYTHONPATH:-}:" != *":${_CANN_LM_SITE}:"* ]]; then
            export PYTHONPATH="${_CANN_LM_SITE}:${PYTHONPATH:-}"
        fi
    else
        _CANN_LM_SITE=""
        echo "[set_env] warning: cann_local_lm 未装于 P1 venv; 跑 bash scripts/build.sh deps" >&2
    fi
fi

echo "[set_env] converter+compile 统一 env (真 CANN, 无 stubs)"
echo "  CANN        : ${ASCEND_HOME_PATH}"
echo "  venv python : ${CONVERTER_PYTHON}"
echo "  dopt python : ${DOPT_PYTHON} (subprocess 用, --quant-method gptq 触发)"
echo "  SOC=${SOC_VERSION}  GE_EXTERNAL_WEIGHT=${GE_EXTERNAL_WEIGHT}  FUSION=$(basename "${GE_FUSION_SWITCH_FILE}")"
echo "  cann_local_lm site: ${_CANN_LM_SITE:-<未发现, 跑 build.sh deps>}"
if [[ "$_MOBILE" -eq 1 ]]; then
    echo "  mobile      : SHARE_OPS=${GE_MOBILE_WEIGHT_SHARE_OPS}"
    echo "  mobile      : ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
    echo "  mobile      : ASCEND_GE_PY_PASS_PATH=${ASCEND_GE_PY_PASS_PATH}"
fi
unset _CANN_LM_SITE _DEV_OP_CPU _DEV_OP_PASS _MOBILE
