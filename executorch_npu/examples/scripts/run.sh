#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Usage: bash examples/scripts/run.sh gemma4|qwen3_5 export-pt2|export-pte|eager|session|mmlu-prepare|mmlu-evaluate|mmlu-report [options]
set -eo pipefail
model="${1:-}"
action="${2:-}"
if [[ "$model" != gemma4 && "$model" != qwen3_5 ]]; then
  echo "model must be gemma4 or qwen3_5" >&2
  exit 2
fi
if [[ $# -lt 2 ]]; then
  echo "usage: $0 gemma4|qwen3_5 ACTION [options]" >&2
  exit 2
fi
shift 2

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u
cd "$EXECUTORCH_NPU_ROOT"
case "$action" in
  export-pt2) module="examples.$model.export.pt2" ;;
  export-pte) module="examples.$model.export.pte" ;;
  eager) module="examples.$model.inference.eager" ;;
  session) module="examples.$model.inference.session" ;;
  mmlu-prepare) module="examples.$model.mmlu.prepare" ;;
  mmlu-evaluate) module="examples.$model.mmlu.evaluate" ;;
  mmlu-report) module="examples.$model.mmlu.report" ;;
  *) echo "unknown action: $action" >&2; exit 2 ;;
esac
exec python -m "$module" "$@"
