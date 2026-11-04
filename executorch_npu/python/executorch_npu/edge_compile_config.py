# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Public Edge compile configuration for supported ExecuTorch APIs."""

from __future__ import annotations

import inspect

from executorch.exir import EdgeCompileConfig


def aten_edge_compile_config() -> EdgeCompileConfig:
    """Return an ATen-compatible configuration for known ExecuTorch APIs."""
    parameters = inspect.signature(EdgeCompileConfig).parameters
    if "_use_edge_ops" in parameters:
        return EdgeCompileConfig(_use_edge_ops=False)
    if "_skip_dim_order" in parameters:
        return EdgeCompileConfig(_skip_dim_order=True)
    return EdgeCompileConfig()
