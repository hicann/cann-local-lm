# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""gemma4 stateful session entry point."""

import os
from pathlib import Path

from examples.common.inference import main as session_main


def main() -> None:
    session_main(
        Path(os.environ.get("MODEL_PATH", "/mnt/workspace/models/gemma-4-E2B-it")),
        Path(os.environ.get("PTE_DIR", "/mnt/workspace/scratch_gemma4_pd/pte")),
        min_prefill_len=2,
        mask_window=512,
        causal_mask=False,
        default_namespace="gemma4_session",
    )


if __name__ == "__main__":
    main()
