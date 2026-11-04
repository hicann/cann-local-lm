#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/examples/mobilenet_v2/artifacts}"
BUILD_DIR="${CANN_BUILD_DIR:-/tmp/executorch_npu-build}"
ACL_SOC_NAME="${ACL_SOC_NAME:-Ascend910_9362}"

find_runner() {
  local name="$1"
  local candidate
  for candidate in \
    "${CANN_RUNNER:-}" \
    "$BUILD_DIR/$name" \
    "$BUILD_DIR/bin/$name" \
    "$BUILD_DIR/install/bin/$name"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  echo "runner not found: $name (checked CANN_BUILD_DIR=$BUILD_DIR)" >&2
  return 1
}

case "${1:-all}" in
  export)
    # 导出与编译不需要 NPU 设备；如需在带设备的机器上走原生 TorchAir 路径，
    # 显式设置 EXECUTORCH_NPU_HOST_ONLY=0 覆盖。
    EXECUTORCH_NPU_HOST_ONLY="${EXECUTORCH_NPU_HOST_ONLY:-1}" \
    python "$ROOT_DIR/examples/mobilenet_v2/export.py" \
      --output-dir "$OUTPUT_DIR" \
      --acl-soc-name "$ACL_SOC_NAME"
    ;;
  run)
    : "${DEVICE_ID:?set DEVICE_ID to the target Ascend device ID}"
    : "${STATE_NAMESPACE:?set STATE_NAMESPACE to a unique request namespace}"
    "$(find_runner executorch_npu_runner)" \
      "$OUTPUT_DIR/model.pte" \
      "$OUTPUT_DIR/ascend.weights.package.ge_aot.ptd" \
      "$OUTPUT_DIR/input.bin" \
      "$OUTPUT_DIR/expected.bin" \
      "$DEVICE_ID" \
      "$STATE_NAMESPACE"
    ;;
  all)
    "$SCRIPT_DIR/run.sh" export
    "$SCRIPT_DIR/run.sh" run
    ;;
  *)
    echo "usage: $0 [export|run|all]" >&2
    exit 2
    ;;
esac
