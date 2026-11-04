# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Keep Gemma4 and Qwen3.5 example entry points aligned."""

import subprocess
import sys
from pathlib import Path


def test_model_example_paths_are_symmetric() -> None:
    examples = Path(__file__).resolve().parents[2] / "examples"
    layouts = []
    for model in ("gemma4", "qwen3_5"):
        folder = examples / model
        layouts.append(
            {
                path.relative_to(folder)
                for path in folder.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            }
        )
    assert layouts[0] == layouts[1]


def test_loading_gemma_config_does_not_load_qwen_model() -> None:
    code = (
        "import sys; import examples.gemma4.model.configuration; "
        "assert 'examples.qwen3_5.model.modeling' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
