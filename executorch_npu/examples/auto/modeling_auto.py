# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Auto model selection for ExecuTorch-NPU causal language models."""

from __future__ import annotations

from collections import OrderedDict
from importlib import import_module
from os import PathLike
from typing import Any

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel

from .configuration_auto import AutoConfig


MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        ("gemma4", "Gemma4ForCausalLM"),
        ("gemma4_text", "Gemma4ForCausalLM"),
        ("qwen3_5", "Qwen3_5ForCausalLM"),
        ("qwen3_5_text", "Qwen3_5ForCausalLM"),
    ]
)

_MODEL_FOR_CAUSAL_LM_MAPPING = {
    "gemma4": ("gemma4", "Gemma4ForCausalLM"),
    "gemma4_text": ("gemma4", "Gemma4ForCausalLM"),
    "qwen3_5": ("qwen3_5", "Qwen3_5ForCausalLM"),
    "qwen3_5_text": ("qwen3_5", "Qwen3_5ForCausalLM"),
}

_CONFIG_LOADING_KWARGS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "revision",
    "subfolder",
    "token",
    "trust_remote_code",
)


def _model_config(config: PreTrainedConfig) -> PreTrainedConfig:
    if config.model_type in ("gemma4", "qwen3_5"):
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            raise ValueError(f"{config.model_type} config does not contain text_config")
        return text_config
    return config


def _model_class(config: PreTrainedConfig) -> type[PreTrainedModel]:
    try:
        model_package, class_name = _MODEL_FOR_CAUSAL_LM_MAPPING[config.model_type]
    except KeyError as error:
        supported = ", ".join(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
        raise ValueError(
            f"Unsupported model type {config.model_type!r}; expected one of: {supported}"
        ) from error
    return getattr(
        import_module(f"examples.{model_package}.model.modeling"), class_name
    )


def _validate_loading_info(loading_info: dict[str, Any]) -> None:
    failures = {
        name: loading_info.get(name)
        for name in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if loading_info.get(name)
    }
    if failures:
        details = "; ".join(f"{name}={value}" for name, value in failures.items())
        raise RuntimeError(f"Model checkpoint did not load strictly: {details}")


class AutoModelForCausalLM:
    """Instantiate a supported local causal language model with HF weights."""

    def __init__(self) -> None:
        raise EnvironmentError(
            "AutoModelForCausalLM must be created with from_config() or from_pretrained()."
        )

    @classmethod
    def from_config(cls, config: PreTrainedConfig, **kwargs: Any) -> PreTrainedModel:
        model_class = _model_class(config)
        return model_class(_model_config(config), **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *model_args: Any,
        config: PreTrainedConfig | None = None,
        strict: bool = True,
        output_loading_info: bool = False,
        **kwargs: Any,
    ) -> PreTrainedModel | tuple[PreTrainedModel, dict[str, Any]]:
        if config is None:
            config_kwargs = {
                name: kwargs[name] for name in _CONFIG_LOADING_KWARGS if name in kwargs
            }
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **config_kwargs,
            )

        model_class = _model_class(config)
        if config.model_type in ("gemma4", "qwen3_5"):
            kwargs.setdefault("key_mapping", {r"^model\.language_model\.": "model."})
        model, loading_info = model_class.from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=_model_config(config),
            output_loading_info=True,
            **kwargs,
        )
        if strict:
            _validate_loading_info(loading_info)
        if output_loading_info:
            return model, loading_info
        return model


__all__ = [
    "AutoModelForCausalLM",
    "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES",
]
