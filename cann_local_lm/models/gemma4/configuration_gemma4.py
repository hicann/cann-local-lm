# Pure Python config classes for Gemma-4 - no transformers dependency.
#
# Based on HuggingFace transformers/src/transformers/models/gemma4/configuration_gemma4.py
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Gemma4AudioConfig:
    """Audio encoder configuration for Gemma-4."""

    model_type: str = "gemma4_audio"
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    hidden_act: str = "silu"

    subsampling_conv_channels: list = field(default_factory=lambda: [128, 32])

    conv_kernel_size: int = 5
    residual_weight: float = 0.5
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1.0e9

    use_clipped_linears: bool = True
    rms_norm_eps: float = 1e-6
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536
    initializer_range: float = 0.02

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class Gemma4TextConfig:
    """Text language model configuration for Gemma-4.

    Default values correspond to google/gemma-4-e2b-it.
    Supports: GQA, QK Norm, Soft-Capping, KV Cache sharing, MoE, Per-Layer Embeddings.
    """

    model_type: str = "gemma4_text"

    vocab_size: int = 262_144
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 512
    final_logit_softcapping: float | None = None
    use_bidirectional_attention: Literal["all", "vision"] | None = None

    vocab_size_per_layer_input: int = 262_144
    hidden_size_per_layer_input: int = 256
    num_global_key_value_heads: int | None = None
    global_head_dim: int = 512
    attention_k_eq_v: bool = False
    num_kv_shared_layers: int = 0
    enable_moe_block: bool = False
    use_double_wide_mlp: bool = False
    num_experts: int | None = None
    top_k_experts: int | None = None
    moe_intermediate_size: int | None = None

    layer_types: list[str] | None = None
    rope_parameters: dict | None = None

    def __post_init__(self):
        if self.layer_types is None:
            sliding_window_pattern = 6
            self.layer_types = [
                "sliding_attention"
                if bool((i + 1) % sliding_window_pattern)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self.layer_types[-1] != "full_attention":
            self.layer_types[-1] = "full_attention"

        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class Gemma4VisionConfig:
    """Vision encoder (ViT) configuration for Gemma-4.

    Features: multi-dimensional (2D) RoPE, QK+V Norm, spatial pooling.
    """

    model_type: str = "gemma4_vision"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    hidden_activation: str = "gelu_pytorch_tanh"
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 131_072
    attention_bias: bool = False
    attention_dropout: float = 0.0
    pooling_kernel_size: int = 3
    patch_size: int = 16
    position_embedding_size: int = 10 * 1024
    use_clipped_linears: bool = False
    standardize: bool = False
    initializer_range: float = 0.02

    rope_parameters: dict | None = None
    layer_types: list[str] | None = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "default", "rope_theta": 100.0}

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class Gemma4Config:
    """Top-level Gemma-4 configuration for multimodal models."""

    model_type: str = "gemma4"

    text_config: Gemma4TextConfig | dict | None = None
    vision_config: Gemma4VisionConfig | dict | None = None
    audio_config: Gemma4AudioConfig | dict | None = None

    boi_token_id: int = 255_999
    eoi_token_id: int = 258_882
    image_token_id: int = 258_880
    video_token_id: int = 258_884
    boa_token_id: int = 256_000
    eoa_token_index: int = 258_883
    audio_token_id: int = 258_881
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = Gemma4TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Gemma4TextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Gemma4VisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = Gemma4VisionConfig(**self.vision_config)

        if self.audio_config is None:
            self.audio_config = Gemma4AudioConfig()
        elif isinstance(self.audio_config, dict):
            self.audio_config = Gemma4AudioConfig(**self.audio_config)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


__all__ = [
    "Gemma4AudioConfig",
    "Gemma4Config",
    "Gemma4TextConfig",
    "Gemma4VisionConfig",
]
