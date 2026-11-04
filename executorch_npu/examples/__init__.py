# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Model implementations, export tools, and runtime examples."""

__all__ = ["AutoConfig", "AutoModelForCausalLM"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from .auto import AutoConfig, AutoModelForCausalLM

    return {"AutoConfig": AutoConfig, "AutoModelForCausalLM": AutoModelForCausalLM}[
        name
    ]
