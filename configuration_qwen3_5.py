"""Qwen3.5 configuration – standalone, no transformers dependencies."""

import json
import os
from typing import Any


class PreTrainedConfig:
    model_type: str = ""
    _auto_class = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__post_init__(**kwargs)

    def __post_init__(self, **kwargs):
        pass

    def to_dict(self):
        d = {}
        for attr in dir(self):
            if not attr.startswith("_") and not callable(getattr(self, attr)):
                try:
                    json.dumps(getattr(self, attr))
                    d[attr] = getattr(self, attr)
                except (TypeError, ValueError):
                    d[attr] = str(getattr(self, attr))
        return d

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(pretrained_model_name_or_path, "config.json")

        with open(config_path) as f:
            config_dict = json.load(f)

        config_dict.update(kwargs)
        return cls(**config_dict)


class Qwen3_5TextConfig(PreTrainedConfig):
    model_type = "qwen3_5_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: Any = None  # set via config.json; must contain "rope_theta"
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]


class Qwen3_5VisionConfig(PreTrainedConfig):
    model_type = "qwen3_5_vision"

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02


class Qwen3_5Config(PreTrainedConfig):
    model_type = "qwen3_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: PreTrainedConfig | None = None
    vision_config: PreTrainedConfig | None = None

    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            if self.vision_config.get("model_type") == "qwen3_5":
                self.vision_config["model_type"] = "qwen3_5_vision"
            self.vision_config = Qwen3_5VisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Qwen3_5VisionConfig()

        if isinstance(self.text_config, dict):
            self.text_config = Qwen3_5TextConfig(**self.text_config)
        elif self.text_config is None:
            self.text_config = Qwen3_5TextConfig()


__all__ = ["Qwen3_5Config", "Qwen3_5TextConfig", "Qwen3_5VisionConfig"]
