#!/bin/bash
# cann_lm set_env.sh — 统一环境设置
#
# 三套独立环境,互不叠加,按需 source:
#   source scripts/set_env.sh converter   # ① converter venv (导出 epair,CPU + torchair stubs, 不装 torch_npu)
#   source scripts/set_env.sh npu         # ② NPU 系统 Python (dopt GPTQ / NPU 出字 / OM 验证, 真 torch_npu + 真 CANN)
#   source scripts/set_env.sh om          # ③ OM 编译 (C++ epair → OM, 真 CANN + SOC + fusion switch)
#
# 默认路径可由环境变量覆盖 (TORCH_DELEGATE_HOME / TORCHAIR_ROOT / DOPT_PYTHON / PROJECT_ROOT /
# CANN_SET_ENV / SOC_VERSION / GE_FUSION_SWITCH_FILE).
#
# 见 AGENTS.md "运行环境设置" 详细说明.

set -u

# ─────────────────── 默认路径 (允许环境变量覆盖) ───────────────────
CANN_SET_ENV="${CANN_SET_ENV:-/home/developer/Ascend/cann/set_env.sh}"
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/workspace/project/CANN-LM_5883}"
TORCHAIR_ROOT="${TORCHAIR_ROOT:-/mnt/workspace/torchair}"
TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME:-/mnt/workspace/torch-delegate-harmony}"
DOPT_PYTHON="${DOPT_PYTHON:-/opt/buildtools/Python-3.11.4/bin/python3}"

# ─────────────────── 公共部分: source CANN set_env.sh (若 ASCEND_HOME_PATH 未设) ───────────────────
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ ! -r "${CANN_SET_ENV}" ]]; then
        echo "[set_env] 错误: CANN set_env.sh 不存在或不可读: ${CANN_SET_ENV}" >&2
        echo "  用 CANN_SET_ENV=... source scripts/set_env.sh <env> 覆盖" >&2
        return 1 2>/dev/null || exit 1
    fi
    source "${CANN_SET_ENV}"
fi

# 公共部分: 项目根加 PYTHONPATH (converter / NPU 都需)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 公共部分: 阻止 torch autoload npu backend (converter venv 不装 torch_npu; NPU 由 torch_npu 显式 import)
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# 公共部分: dopt subprocess 用系统 Python 路径 (auto_quant.py:83 读此变量, fallback 到 /opt/buildtools/Python-3.11.4/bin/python3)
export DOPT_PYTHON="${DOPT_PYTHON}"

# ─────────────────── 按环境选择 ───────────────────
case "${1:-converter}" in
    converter)
        # ① converter venv: torchair stubs + 不装 torch_npu (走 stub 路径)
        #    用于: 导出 epair (torch.export + torchair GE 转换)
        export TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME}"
        export TORCH_DELEGATE_ENABLE_FALLBACK_CODEGEN=0
        export ASCEND_SLOG_PRINT_TO_STDOUT=1   # GE 详细日志 (定位 not registered 等)
        # stubs 必须在 CANN lib64 前 (否则 aclrtSetDevice → 107001, 无设备)
        export LD_LIBRARY_PATH="${TORCHAIR_ROOT}/build/stubs:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
        # converter venv python (跑 cann_lm_converter.py / tests/)
        export CONVERTER_PYTHON="${TORCH_DELEGATE_HOME}/.venv/bin/python"
        echo "[set_env] ① converter venv (CPU + torchair stubs, 不装 torch_npu)"
        echo "  venv python : ${CONVERTER_PYTHON}"
        echo "  dopt python : ${DOPT_PYTHON} (subprocess 用, --quant-method gptq 触发)"
        echo "  stubs       : ${TORCHAIR_ROOT}/build/stubs"
        ;;
    npu)
        # ② NPU 系统 Python: 真 CANN lib64 + torch_npu, 无 stubs
        #    用于: dopt GPTQ / NPU 出字 / OM 验证 (run_*_om.py 等)
        # 关键差异: 不设 stubs (与 converter 区分); 不设 TORCH_DELEGATE_*
        export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
        # DEVICE 由 set_env.sh 统一 export:converter --quant-method gptq 时 subprocess 继承此变量,
        # 不再硬编码覆盖 (auto_quant.py:_run_dopt_subprocess 用 env.setdefault("DEVICE", device))
        export DEVICE=npu
        echo "[set_env] ② NPU system Python (真 torch_npu + 真 CANN)"
        echo "  python      : ${DOPT_PYTHON}"
        echo "  lib64       : ${ASCEND_HOME_PATH}/lib64 (无 stubs)"
        echo "  DEVICE      : ${DEVICE}"
        ;;
    om)
        # ③ OM 编译: 真 CANN + SOC + fusion switch (关 AUB fusion)
        #    用于: epair → OM 编译 (export_llm_model C++ 二进制)
        export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
        export SOC_VERSION="${SOC_VERSION:-Ascend910_9362}"
        # 关 fusion: AUB (AutomaticUbFusion) + AddRmsNorm 残差融合
        # (AddRmsNormFusionGraphPass / InplaceAddRmsNormFusionPass). 关后者使 OM 保留分开的
        # RmsNorm+Add (ascend_ir 原样), 不产生 InplaceAddRmsNorm 融合算子 (未在 support_ops.md 列出).
        # 如需该融合, 删 GraphFusion 两项即可 (融合后 OM 少 35 个算子但 InplaceAddRmsNorm 未覆盖).
        GE_FUSION_SWITCH_FILE="${GE_FUSION_SWITCH_FILE:-/tmp/fusion_autoub_off.cfg}"
        echo '{"Switch":{"UBFusion":{"AutomaticUbFusion":"off"},"GraphFusion":{"AddRmsNormFusionGraphPass":"off","InplaceAddRmsNormFusionPass":"off"}}}' > "${GE_FUSION_SWITCH_FILE}"
        export GE_FUSION_SWITCH_FILE
        echo "[set_env] ③ OM compile (C++ epair → OM)"
        echo "  SOC_VERSION         : ${SOC_VERSION}"
        echo "  GE_FUSION_SWITCH_FILE: ${GE_FUSION_SWITCH_FILE}"
        ;;
    *)
        echo "Usage: source scripts/set_env.sh [converter|npu|om]"
        echo "  converter (默认): 导出 epair (CPU venv + torchair stubs)"
        echo "  npu:             dopt GPTQ / NPU 出字 (系统 Python + torch_npu)"
        echo "  om:               OM 编译 (C++ 二进制 + SOC + fusion switch)"
        echo ""
        echo "环境变量覆盖 (可选):"
        echo "  CANN_SET_ENV          CANN set_env.sh 路径 (默认 /home/developer/Ascend/cann/set_env.sh)"
        echo "  PROJECT_ROOT          项目根路径 (默认 /mnt/workspace/project/CANN-LM_5883)"
        echo "  TORCHAIR_ROOT         torchair 源码根 (默认 /mnt/workspace/torchair)"
        echo "  TORCH_DELEGATE_HOME   torch-delegate 安装路径 (默认 /mnt/workspace/torch-delegate-harmony)"
        echo "  DOPT_PYTHON           dopt subprocess 用的系统 Python (默认 /opt/buildtools/Python-3.11.4/bin/python3)"
        echo "  SOC_VERSION           OM 编译目标 SOC (默认 Ascend910_9362)"
        return 1 2>/dev/null || exit 1
        ;;
esac

set +u
