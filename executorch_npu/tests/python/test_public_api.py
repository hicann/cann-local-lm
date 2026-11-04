# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import subprocess
import sys


def test_public_partitioner_import_registers_backend() -> None:
    code = """
from executorch.exir.backend.backend_details import BackendDetails
from executorch_npu import AscendPartitioner

assert AscendPartitioner.__name__ == "AscendPartitioner"
assert "AscendBackend" in {cls.__name__ for cls in BackendDetails.__subclasses__()}
"""
    subprocess.run([sys.executable, "-c", code], check=True)
