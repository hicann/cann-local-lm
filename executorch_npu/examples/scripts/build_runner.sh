#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Build runners using the active Python environment and selected CANN install.
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit $?
set -u
: "${ASCEND_HOME_PATH:?Load CANN first or set CANN_ENV_SCRIPT}"
cd "$EXECUTORCH_NPU_ROOT" || exit $?
EXECUTORCH_SDK="${EXECUTORCH_SDK:-$(python - <<'PYTHON'
import importlib.util
spec = importlib.util.find_spec("executorch")
if not spec or not spec.submodule_search_locations:
    raise SystemExit("Install executorch in the active Python environment or set EXECUTORCH_SDK")
print(next(iter(spec.submodule_search_locations)))
PYTHON
)}" || exit $?
CANN_ET_BUILD_DIR="${CANN_ET_BUILD_DIR:-$EXECUTORCH_NPU_ROOT/build/release}"
CANN_ET_BUILD_JOBS="${CANN_ET_BUILD_JOBS:-8}"

echo "[$(date "+%F %T")] runner build start"
echo "  ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "  EXECUTORCH_SDK=$EXECUTORCH_SDK"
echo "  BUILD_DIR=$CANN_ET_BUILD_DIR"

cmake -S . -B "$CANN_ET_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CANN_ET_BUILD_DIR/install" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DEXECUTORCH_INCLUDE_DIR="$EXECUTORCH_SDK/include" \
  -Dexecutorch_DIR="$EXECUTORCH_SDK/share/cmake" \
  -DCMAKE_PREFIX_PATH="$EXECUTORCH_SDK" \
  -DEXECUTORCH_NPU_BUILD_RUNNER=ON
configure_status=$?
echo "[$(date "+%F %T")] CONFIGURE_EXIT=$configure_status"
if [ "$configure_status" -ne 0 ]; then
  echo "[$(date "+%F %T")] BUILD_EXIT=$configure_status"
  exit "$configure_status"
fi

cmake --build "$CANN_ET_BUILD_DIR" -j"$CANN_ET_BUILD_JOBS" \
  --target executorch_npu_stateful_runner executorch_npu_runner
build_status=$?
echo "[$(date "+%F %T")] BUILD_EXIT=$build_status"
ls -la "$CANN_ET_BUILD_DIR/executorch_npu_stateful_runner" 2>&1
exit "$build_status"
