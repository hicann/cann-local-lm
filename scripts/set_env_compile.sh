#!/bin/bash
# set_env_compile.sh — ② compile env (epair→OM: C++ 编译 + OM 编译)
#   真 CANN lib64 + SOC + fusion switch (关 AUB fusion)
#   工具不依赖 torch/venv (csrc 零 torch 引用), 构建/运行只需 CANN
#
# 用法: source scripts/set_env_compile.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env_common.sh"

export TORCH_DELEGATE_HOME="${TORCH_DELEGATE_HOME}"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export SOC_VERSION="${SOC_VERSION:-Ascend910_9362}"
# 关 fusion: AUB (AutomaticUbFusion) + AddRmsNorm 残差分支融合
# 静态配置文件 (csrc/tools/fusion_autoub_off.cfg), 直接读取, 不再每次 echo 生成到 /tmp
export GE_FUSION_SWITCH_FILE="${GE_FUSION_SWITCH_FILE:-${PROJECT_ROOT}/cann_lm/converter/export/csrc/tools/fusion_autoub_off.cfg}"
# 权值分离: 0=内嵌 / 1=分文件 / 2=全合一 .bin (默认 2: .om 仅结构, 权值存 <output 目录>/weight/)
# 约束: output 路径须与 CWD 同文件系统 (GE 临时权值在 CWD, MoveFilePath 跨 fs rename 会 EXDEV 失败)
export GE_EXTERNAL_WEIGHT="${GE_EXTERNAL_WEIGHT:-2}"

echo "[set_env] ② compile (epair→OM: C++ 编译 + OM 编译, 真 CANN, 无 torch/venv)"
echo "  lib64               : ${ASCEND_HOME_PATH}/lib64 (无 stubs)"
echo "  SOC_VERSION         : ${SOC_VERSION}"
echo "  GE_FUSION_SWITCH_FILE: ${GE_FUSION_SWITCH_FILE}"
echo "  GE_EXTERNAL_WEIGHT : ${GE_EXTERNAL_WEIGHT} (0=内嵌/1=分文件/2=全合一; output 须与 CWD 同 fs)"
