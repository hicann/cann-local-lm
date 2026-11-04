#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Install MMLU preparation dependencies into the active Python environment.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit $?
set -uo pipefail
echo "[$(date "+%F %T")] pip install start -> $(which python)"
python -m pip install --no-deps datasets pyarrow dill multiprocess xxhash huggingface_hub "lm-eval==0.4.13"
status=$?
echo "[$(date "+%F %T")] PIP_EXIT=$status"
python - <<"PY"
import importlib
for m in ["datasets", "huggingface_hub", "pyarrow", "dill", "multiprocess", "xxhash", "lm_eval"]:
    try:
        mod = importlib.import_module(m)
        print("OK", m, getattr(mod, "__version__", ""))
    except Exception as exc:
        print("FAIL", m, type(exc).__name__, str(exc)[:120])
PY
echo "[$(date "+%F %T")] VERIFY_DONE"
