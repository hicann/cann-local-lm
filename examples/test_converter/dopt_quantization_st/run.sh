#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 从已安装的 .run 包设置环境
CANN_DEVICE_LM="${CANN_DEVICE_LM:-/opt/cann-device-lm}"
source "$HOME/Ascend/cann/set_env.sh"
source "${CANN_DEVICE_LM}/set_env.sh" --npu

PYTHON_BIN="${DOPT_PYTHON:-python3}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_st.py" "$@"
