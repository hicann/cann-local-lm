#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Optionally load user-provided network setup before downloading MMLU.
set -eo pipefail
if [[ -n "${CANN_ET_BOOTSTRAP_SCRIPT:-}" ]]; then
  source "$CANN_ET_BOOTSTRAP_SCRIPT"
fi
exec bash "$(dirname "${BASH_SOURCE[0]}")/download_mmlu.sh"
