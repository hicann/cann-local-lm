#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402 - registers torch.npu and native operators

from examples.auto import AutoModelForCausalLM  # noqa: E402


MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/workspace/models/gemma-4-E2B-it")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "4"))
DEVICE = torch.device(os.environ.get("DEVICE", "npu:0"))
DTYPE = torch.bfloat16


def main() -> None:
    torch.npu.set_device(DEVICE)
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=DTYPE,
            local_files_only=True,
            low_cpu_mem_usage=True,
            strict=True,
            attn_implementation="eager",
        )
        .to(DEVICE)
        .eval()
    )
    model.requires_grad_(False)

    input_ids = torch.zeros((1, SEQ_LEN), dtype=torch.long, device=DEVICE)
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids.unsqueeze(0),
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    print(f"GEMMA4_SINGLE_OP_OK shape={tuple(output.logits.shape)}")


if __name__ == "__main__":
    main()
