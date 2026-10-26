#!/usr/bin/env bash
# 算子级 NPU 验证 — Step 2: cann_lm_compile epair → OM (CANN env: source scripts/cann_lm_compile/set_env_cann.sh)
set -euo pipefail
EPAIR_DIR="${1:?usage: compile_om.sh <epair_dir>}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
COMPILE_BIN="${PROJECT_ROOT}/build/cann_lm_compile"
[[ -x "${COMPILE_BIN}" ]] || { echo "[compile_om] cann_lm_compile 未构建: ${COMPILE_BIN} (跑 bash scripts/build.sh cpp)"; exit 1; }
# cann_lm_compile 退出时 GE teardown 可能 abort, 用 || true 容错, 以 .om 存在判定
"${COMPILE_BIN}" compile --ep-path "${EPAIR_DIR}/op_model.pt2" --output-dir "${EPAIR_DIR}" || true
OM="${EPAIR_DIR}/op_model.om"
if [[ -f "${OM}" ]]; then
  echo "[compile_om] ${OM} ($(ls -lh "${OM}" | awk '{print $5}'))"
else
  echo "[compile_om] FAIL: OM 未生成 (${EPAIR_DIR})"; exit 1
fi
