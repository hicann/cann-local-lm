# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from .base import NodeIR, NodeIRConverter

__all__ = [
    "NodeIR",
    "NodeIRConverter",
    "TorchAirGEConverter",
]


def __getattr__(name: str):
    """Load the TorchAir converter only after the export mode is selected."""
    if name == "TorchAirGEConverter":
        from ..lite_context import active_config, mark_full_initialized

        if active_config() is None:
            mark_full_initialized()
        from .torchair_converter import TorchAirGEConverter

        return TorchAirGEConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
