#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Local convenience defaults; all environment paths can be overridden.
export PATH="${CANN_ET_PYTHON_BIN:-/mnt/workspace/miniforge3/envs/executorch_npu/bin}:$PATH"
export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/mnt/workspace/Ascend/cann-9.1.0/set_env.sh}"
source examples/scripts/env.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
exec python -u -m examples.common.light_eval "$@"
