# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Backend identity and manifest protocol constants."""

from typing import Final


BACKEND_ID: Final[str] = "AscendBackend"
FRAMING_MAGIC: Final[bytes] = b"ASCD"
FRAMING_VERSION: Final[int] = 1
PTD_FILE_TAG: Final[str] = "ascend.weights.package.ge_aot"
WEIGHT_ALIGNMENT: Final[int] = 32
MAX_MANIFEST_BYTES: Final[int] = 1024 * 1024
