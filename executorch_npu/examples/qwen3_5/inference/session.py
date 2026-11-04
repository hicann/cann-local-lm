# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""qwen3_5 stateful session entry point."""

import os
from pathlib import Path

from examples.common.inference import main as session_main


def main() -> None:
    session_main(
        Path(os.environ.get("MODEL_PATH", "/mnt/workspace/models/Qwen3.5-0.8B")),
        Path(os.environ.get("PTE_DIR", "/tmp/qwen3_5_executorch_npu")),
        min_prefill_len=4,
        mask_window=None,
        causal_mask=True,
        default_namespace="qwen3_5_session",
    )


if __name__ == "__main__":
    main()
