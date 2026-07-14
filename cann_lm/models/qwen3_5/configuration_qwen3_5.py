from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields


@dataclass
class Qwen3_5TextConfig:
    model_type: str = "qwen3_5_text"
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0
    attention_bias: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # Q Gate: full-attention layers emit 2x q_proj (query + gate, sigmoid-modulated).
    # Qwen3.5-0.8B sets attn_output_gate=true.
    attn_output_gate: bool = True

    # Multi-token prediction head (present in the VLM checkpoint; the standalone
    # text CausalLM does not use it, but the field is kept so from_pretrained does
    # not silently drop it).
    mtp_num_hidden_layers: int = 0
    mtp_use_dedicated_embeddings: bool = False
    mamba_ssm_dtype: str = "float32"
    mlp_only_layers: list = field(default_factory=list)

    layer_types: list[str] | None = None
    full_attention_interval: int = 4

    rope_parameters: dict = field(default_factory=lambda: {
        "rope_type": "default",
        "rope_theta": 1000000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    })

    dtype: str | None = None

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % self.full_attention_interval)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3_5TextConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        obj = cls(**filtered)
        return obj

    @classmethod
    def from_json_file(cls, path: str) -> Qwen3_5TextConfig:
        with open(path) as f:
            data = json.load(f)
        if "text_config" in data:
            data = data["text_config"]
        return cls.from_dict(data)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> Qwen3_5TextConfig:
        if os.path.isdir(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(model_name_or_path, "config.json")
        return cls.from_json_file(config_path)


@dataclass
class Qwen3_5VisionConfig:
    model_type: str = "qwen3_5_vision"
    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, d: dict) -> Qwen3_5VisionConfig:
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class Qwen3_5Config:
    model_type: str = "qwen3_5"
    text_config: Qwen3_5TextConfig | dict | None = None
    vision_config: Qwen3_5VisionConfig | dict | None = None
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = Qwen3_5TextConfig.from_dict(self.text_config)
        elif self.text_config is None:
            self.text_config = Qwen3_5TextConfig()

        if isinstance(self.vision_config, dict):
            self.vision_config = Qwen3_5VisionConfig.from_dict(self.vision_config)
        elif self.vision_config is None:
            self.vision_config = Qwen3_5VisionConfig()

    @classmethod
    def from_json_file(cls, path: str) -> Qwen3_5Config:
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> Qwen3_5Config:
        if os.path.isdir(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(model_name_or_path, "config.json")
        return cls.from_json_file(config_path)
