# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Auto configuration selection for supported ExecuTorch-NPU models."""

from __future__ import annotations

from collections import OrderedDict
from importlib import import_module
from os import PathLike
from typing import Any

from transformers.configuration_utils import PreTrainedConfig


CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("gemma4", "Gemma4Config"),
        ("gemma4_text", "Gemma4TextConfig"),
        ("qwen3_5", "Qwen3_5Config"),
        ("qwen3_5_text", "Qwen3_5TextConfig"),
    ]
)

_CONFIG_MAPPING = {
    "gemma4": ("gemma4", "Gemma4Config"),
    "gemma4_text": ("gemma4", "Gemma4TextConfig"),
    "qwen3_5": ("qwen3_5", "Qwen3_5Config"),
    "qwen3_5_text": ("qwen3_5", "Qwen3_5TextConfig"),
}


class AutoConfig:
    """Load a Transformers config accepted by a local model implementation."""

    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoConfig must be created with AutoConfig.from_pretrained()."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        **kwargs: Any,
    ) -> PreTrainedConfig:
        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        kwargs.pop("trust_remote_code", None)
        config_dict, unused_kwargs = PreTrainedConfig.get_config_dict(
            pretrained_model_name_or_path,
            **kwargs,
        )
        model_type = config_dict.get("model_type")
        if model_type not in _CONFIG_MAPPING:
            supported = ", ".join(CONFIG_MAPPING_NAMES)
            raise ValueError(
                f"Unsupported model type {model_type!r}; expected one of: {supported}"
            )
        model_package, class_name = _CONFIG_MAPPING[model_type]
        config_class = getattr(
            import_module(f"examples.{model_package}.model.configuration"), class_name
        )
        return config_class.from_dict(config_dict, **unused_kwargs)


__all__ = ["AutoConfig", "CONFIG_MAPPING_NAMES"]
