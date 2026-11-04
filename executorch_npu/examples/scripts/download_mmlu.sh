#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Robustly cache the cais/mmlu parquet files (the Hub proxy drops connections).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" || exit $?
set -uo pipefail

python - <<"PY"
import time
from huggingface_hub import snapshot_download

patterns = [
    "*/dev-00000-of-00001.parquet",
    "*/test-00000-of-00001.parquet",
    "*/validation-00000-of-00001.parquet",
    "dataset_infos.json",
    "README.md",
]
last_error = None
for attempt in range(1, 9):
    try:
        path = snapshot_download(
            "cais/mmlu",
            repo_type="dataset",
            allow_patterns=patterns,
            max_workers=4,
            etag_timeout=60,
        )
        print("SNAPSHOT_OK", path)
        break
    except Exception as exc:  # noqa: BLE001 - network flakiness
        last_error = exc
        print(f"SNAPSHOT_RETRY attempt={attempt} error={type(exc).__name__}: {exc}", flush=True)
        time.sleep(min(2 ** attempt, 30))
else:
    raise SystemExit(f"snapshot_download failed: {last_error}")
PY
status=$?
echo "[$(date "+%F %T")] DOWNLOAD_EXIT=$status"
exit "$status"
