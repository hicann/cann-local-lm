"""Model registry — maps model names to config/model/weight-loader entries."""

from __future__ import annotations

import dataclasses
from typing import Callable

from cann_lm.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from cann_lm.models.gemma4.modeling_gemma4 import load_gemma4_weights
from cann_lm.models.qwen3 import Qwen3ForCausalLM, Qwen3TextConfig
from cann_lm.models.qwen3.modeling_qwen3 import load_qwen3_weights
from cann_lm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from cann_lm.models.qwen3_5.modeling_qwen3_5 import load_qwen3_5_weights


@dataclasses.dataclass
class ModelEntry:
    model_type: str
    config_cls: type
    model_cls: type
    load_weights_fn: Callable
    config_json_key: str


REGISTRY: dict[str, ModelEntry] = {
    "gemma4": ModelEntry(
        model_type="gemma4_text",
        config_cls=Gemma4TextConfig,
        model_cls=Gemma4ForCausalLM,
        load_weights_fn=load_gemma4_weights,
        config_json_key="text_config",
    ),
    "qwen3_5": ModelEntry(
        model_type="qwen3_5_text",
        config_cls=Qwen3_5TextConfig,
        model_cls=Qwen3_5ForCausalLM,
        load_weights_fn=load_qwen3_5_weights,
        config_json_key="text_config",
    ),
    "qwen3": ModelEntry(
        model_type="qwen3",
        config_cls=Qwen3TextConfig,
        model_cls=Qwen3ForCausalLM,
        load_weights_fn=load_qwen3_weights,
        config_json_key="text_config",
    ),
}

_ALIASES: dict[str, str] = {
    "gemma4": "gemma4",
    "gemma-4": "gemma4",
    "qwen3.5": "qwen3_5",
    "qwen3_5": "qwen3_5",
    "qwen3": "qwen3",
    "qwen-3": "qwen3",
}


def get_entry(name: str) -> ModelEntry:
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in REGISTRY:
        raise ValueError(f"Unknown model '{name}', available: {list(REGISTRY)}")
    return REGISTRY[key]


def available_models() -> list[str]:
    return list(REGISTRY)
