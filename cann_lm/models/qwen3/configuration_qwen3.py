"""Pure Python config class for Qwen3 (standard, non-Next) - no transformers dependency.

Based on HuggingFace transformers/src/transformers/models/qwen3/configuration_qwen3.py.
Standard Qwen3 is the base of Qwen3.5: plain GQA + QK-norm + SwiGLU + default RoPE,
without the Q Gate / linear attention / MRoPE / centered RMSNorm of Qwen3.5.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields


@dataclass
class Qwen3TextConfig:
    """Text language model configuration for standard Qwen3.

    Default values correspond to Qwen/Qwen3-0.6B.
    """

    model_type: str = "qwen3"
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0
    attention_bias: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    rope_theta: float = 1_000_000.0
    rope_scaling: dict | None = None
    sliding_window: int | None = None
    use_sliding_window: bool = False
    max_window_layers: int | None = None

    dtype: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3TextConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str) -> Qwen3TextConfig:
        with open(path) as f:
            data = json.load(f)
        if "text_config" in data and isinstance(data["text_config"], dict):
            data = data["text_config"]
        return cls.from_dict(data)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> Qwen3TextConfig:
        if os.path.isdir(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(model_name_or_path, "config.json")
        return cls.from_json_file(config_path)


__all__ = ["Qwen3TextConfig"]
