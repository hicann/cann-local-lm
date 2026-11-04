# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Lazy public exports for the qwen3_5 example."""

from importlib import import_module


__all__ = [
    "Qwen3_5Config",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5Model",
    "Qwen3_5PreTrainedModel",
    "Qwen3_5TextConfig",
    "Qwen3_5TextModel",
    "Qwen3_5VisionConfig",
    "Qwen3_5VisionModel",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = "model.configuration" if name.endswith("Config") else "model.modeling"
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
