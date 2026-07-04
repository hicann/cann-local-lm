"""Gemma-4 pure PyTorch implementation - no transformers dependency.

Based on HuggingFace transformers/src/transformers/models/gemma4/modeling_gemma4.py
Copyright 2026 the HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0.

See the SKILL.md for architecture overview and design decisions.
"""

import math
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4Config,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)


# ════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (for use in both text and vision)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    if x_pass.shape[-1] > 0:
        return torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)
    return (x * cos) + (rotate_half(x) * sin)

def apply_rotary_pos_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Handle partial rotary: only rotate rotary_dim channels
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    if q_pass.shape[-1] > 0:
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: (b, kv_heads, s, d) -> (b, heads, s, d)."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    softcap: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager (non-flash) scaled dot-product attention with optional soft-capping."""
    if scaling is None:
        scaling = module.head_dim ** -0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ════════════════════════════════════════════════════════════════════
# RMSNorm
# ════════════════════════════════════════════════════════════════════

class Gemma4RMSNorm(nn.Module):
    """RMSNorm matching the Gemma-4 convention (pow(-0.5) instead of rsqrt)."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = x.pow(2).mean(-1, keepdim=True) + self.eps
        return x * torch.pow(mean_sq, -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        if self.with_scale:
            output = output * self.weight.float()
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape) if self.with_scale else '()'}, eps={self.eps}"

class Gemma4TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None, embed_scale: float | None = None):
        if embed_scale is None:
            embed_scale = math.sqrt(embedding_dim)
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.register_buffer("embed_scale", torch.tensor(embed_scale, dtype=torch.float32), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)

# ════════════════════════════════════════════════════════════════════
# Rotary Embeddings
# ════════════════════════════════════════════════════════════════════

class Gemma4TextRotaryEmbedding(nn.Module):
    """Per-layer-type RoPE for Gemma-4 text model.

    Each layer_type (full_attention, sliding_attention) can have different
    rope_type (default, proportional), rope_theta, and head_dim.
    """

    def __init__(self, config: Gemma4TextConfig, device=None):
        nn.Module.__init__(self)
        self.config = config
        self.layer_types = set(config.layer_types)
        self._rope_configs: dict[str, dict] = {}

        for layer_type in self.layer_types:
            rope_params = config.rope_parameters.get(layer_type)
            if rope_params is None:
                continue

            rope_type = rope_params.get("rope_type", "default")
            base = rope_params.get("rope_theta", 10_000.0)
            partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)

            if layer_type == "full_attention" and rope_type == "proportional":
                head_dim = config.global_head_dim
            else:
                head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

            self._rope_configs[layer_type] = {
                "rope_type": rope_type,
                "base": base,
                "partial_rotary_factor": partial_rotary_factor,
                "head_dim": head_dim,
            }

    def _compute_inv_freq(self, layer_type: str, device: torch.device) -> torch.Tensor:
        rope_config = self._rope_configs[layer_type]
        rope_type = rope_config["rope_type"]
        base = rope_config["base"]
        partial_rotary_factor = rope_config["partial_rotary_factor"]
        head_dim = rope_config["head_dim"]

        if rope_type == "proportional":
            rope_angles = int(partial_rotary_factor * head_dim // 2)
            inv_freq_rotated = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim))
            nope_angles = head_dim // 2 - rope_angles
            if nope_angles > 0:
                return torch.cat((inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32, device=device)), dim=0)
            else:
                return inv_freq_rotated
        else:
            dim = int(head_dim * partial_rotary_factor)
            return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str = "full_attention"):
        inv_freq = self._compute_inv_freq(layer_type, x.device)
        attn_scaling = 1.0

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attn_scaling
        sin = emb.sin() * attn_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ════════════════════════════════════════════════════════════════════
# Text Model Components
# ════════════════════════════════════════════════════════════════════

class Gemma4TextMLP(nn.Module):
    """Gated MLP (SwiGLU-style) for Gemma-4 text decoder.

    KV-shared layers may use double-wide intermediate size when
    `use_double_wide_mlp` is enabled.
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared > 0
        double_wide = config.use_double_wide_mlp and is_kv_shared_layer
        self.intermediate_size = config.intermediate_size * (2 if double_wide else 1)
        self.hidden_size = config.hidden_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma4TextAttention(nn.Module):
    """Multi-headed attention for Gemma-4 text decoder.

    Key features:
    - QK normalization
    - Different head_dim for global vs sliding attention
    - KV sharing: last N layers can share KV from earlier layer
    - Alternative attention (K=V): global layers may reuse K as V
    - scaling = 1.0 (no head_dim^-0.5)
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = (
            config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        )
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        num_kv_heads = (
            config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads
        )
        self.num_key_value_groups = config.num_attention_heads // num_kv_heads
        self.scaling = 1.0
        self.attention_dropout = config.attention_dropout
        self.is_causal = config.use_bidirectional_attention != "all"

        # KV sharing
        first_kv_shared = config.num_hidden_layers - getattr(config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared >= 0
        prev_types = config.layer_types[:first_kv_shared]
        self.store_full_length_kv = (
            not self.is_kv_shared_layer
            and layer_idx == len(prev_types) - 1 - prev_types[::-1].index(config.layer_types[layer_idx])
        )

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
            self.k_proj = nn.Linear(
                config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = (
                nn.Linear(config.hidden_size, num_kv_heads * self.head_dim, bias=config.attention_bias)
                if not self.use_alternative_attention
                else None
            )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        shared_kv_states: dict | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        # Q uses num_attention_heads, k/v use num_key_value_heads
        q_shape = (*input_shape, self.config.num_attention_heads, self.head_dim)
        num_kv_heads = self.config.num_key_value_heads
        kv_shape = (*input_shape, num_kv_heads, self.head_dim)

        # --- Q projection ---
        query_states = self.q_proj(hidden_states).view(q_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # --- KV projection (or reuse shared) ---
        if self.is_kv_shared_layer and shared_kv_states is not None:
            kv_key = self.layer_type
            if kv_key not in shared_kv_states:
                kv_key = next(iter(shared_kv_states), None)
            if kv_key is not None:
                key_states, value_states = shared_kv_states[kv_key]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(kv_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            if self.v_proj is not None:
                value_states = self.v_proj(hidden_states).view(kv_shape)
            else:
                value_states = key_states
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        # --- KV cache update ---
        if past_key_values is not None and not self.is_kv_shared_layer:
            if past_key_values[self.layer_idx] is not None:
                key_states = torch.cat([past_key_values[self.layer_idx][0], key_states], dim=2)
                value_states = torch.cat([past_key_values[self.layer_idx][1], value_states], dim=2)
            past_key_values[self.layer_idx] = (key_states, value_states)

        if self.store_full_length_kv and shared_kv_states is not None:
            shared_kv_states[self.layer_type] = (key_states, value_states)

        # --- Attention ---
        attn_output, _attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


# ════════════════════════════════════════════════════════════════════
# Decoder Layer and Model
# ════════════════════════════════════════════════════════════════════

class Gemma4TextDecoderLayer(nn.Module):
    """Single decoder layer with pre/post norms for both attention and FFN.

    Norm layout matches Gemma-3/Gemma-4 convention:
      input -> input_layernorm -> attn -> post_attn_layernorm -> (+ residual)
            -> pre_ffn_layernorm -> mlp -> post_ffn_layernorm -> (+ residual)
            -> per_layer_input_gate -> per_layer_projection -> post_per_layer_input_norm -> (+ residual)
            -> * layer_scalar
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.self_attn = Gemma4TextAttention(config, layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = nn.Linear(config.hidden_size, config.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(config.hidden_size_per_layer_input, config.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def _compute_per_layer_input(
        self, 
        per_layer_embeds: torch.Tensor,
        per_layer_model_projection: nn.Linear,
        per_layer_projection_norm: Gemma4RMSNorm,
    ) -> torch.Tensor:
        projected = per_layer_model_projection(per_layer_embeds.float())
        projected = per_layer_projection_norm(projected)
        return projected

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
        shared_kv_states: dict | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            shared_kv_states=shared_kv_states,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            gated = self.per_layer_input_gate(hidden_states)
            gelued = F.gelu(gated, approximate="tanh")
            multiplied = gelued * per_layer_input
            projected = self.per_layer_projection(multiplied)
            normed = self.post_per_layer_input_norm(projected)
            hidden_states = residual + normed
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4TextModel(nn.Module):
    """Pure PyTorch implementation of the Gemma-4 Text Model (decoder-only)."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = Gemma4TextScaledWordEmbedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            ple_vocab_size = config.vocab_size_per_layer_input
            ple_embed_dim = config.num_hidden_layers * config.hidden_size_per_layer_input
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                ple_vocab_size, ple_embed_dim, config.pad_token_id,
                embed_scale=config.hidden_size_per_layer_input ** 0.5,
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size, config.num_hidden_layers * config.hidden_size_per_layer_input, bias=False
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)
    
    @property
    def per_layer_input_scale(self) -> float:
        return 2.0**-0.5
    
    @property
    def per_layer_model_projection_scale(self) -> float:
        return self.config.hidden_size**-0.5
    
    def _prepare_position_ids(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: list | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = inputs_embeds.shape[:2]
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
        return torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device).unsqueeze(0)

    def _prepare_causal_mask(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        layer_type: str = "sliding_attention",
    ) -> torch.Tensor | None:
        batch_size, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        min_dtype = torch.finfo(dtype).min

        q_pos = torch.arange(seq_len, device=device)
        key_pos = torch.arange(seq_len, device=device)
        causal_mask = key_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

        if attention_mask is None:
            if layer_type == "full_attention":
                full_mask = causal_mask
            elif layer_type == "sliding_attention":
                sliding_window = getattr(self.config, "sliding_window", None)
                if sliding_window is None:
                    return None
                diff = q_pos.unsqueeze(1) - key_pos.unsqueeze(0)
                window_mask = (diff >= 0) & (diff <= sliding_window - 1)
                full_mask = causal_mask & window_mask
            else:
                full_mask = causal_mask
            mask = full_mask
            return torch.where(mask, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(min_dtype, device=device, dtype=dtype)).unsqueeze(0).unsqueeze(0)

        if attention_mask.dim() == 2:
            attention_mask = attention_mask.to(dtype=torch.bool)
            key_valid = attention_mask[:, None, :] # [batch, 1, seq_len]
            if layer_type == "full_attention":
                full_mask = key_valid.expand(batch_size, seq_len, seq_len) & causal_mask
            elif layer_type == "sliding_attention":
                sliding_window = getattr(self.config, "sliding_window", None)
                if sliding_window is not None:
                    diff = q_pos.unsqueeze(1) - key_pos.unsqueeze(0)
                    window_mask = (diff >= 0) & (diff <= sliding_window - 1)
                    full_mask = (causal_mask & window_mask).unsqueeze(0) & key_valid 
                else:
                    full_mask = causal_mask.unsqueeze(0) & key_valid
            else:
                if attention_mask.all():
                    return None
                full_mask = key_valid.expand(batch_size, seq_len, seq_len)
            
            mask = full_mask
            return torch.where(mask, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(min_dtype, device=device, dtype=dtype)).unsqueeze(0)

        return attention_mask

    def _build_rope_embeddings_for_layers(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Pre-compute RoPE cos/sin for each layer type."""
        rope_embs = {}
        for layer_type in set(self.config.layer_types):
            rope_embs[layer_type] = self.rotary_emb(inputs_embeds, position_ids, layer_type)
        return rope_embs

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        inputs_embeds: torch.Tensor | None = None,
        shared_kv_states: dict | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list, dict | None]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        if position_ids is None:
            position_ids = self._prepare_position_ids(inputs_embeds, past_key_values)

        causal_mask = self._prepare_causal_mask(inputs_embeds, attention_mask)
        rope_embs = self._build_rope_embeddings_for_layers(inputs_embeds, position_ids)

        if shared_kv_states is None:
            shared_kv_states = {}

        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = self._compute_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self._project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            pos_emb = rope_embs[layer_type]

            mask = self._prepare_causal_mask(inputs_embeds, attention_mask, layer_type)

            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden_states = layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=pos_emb,
                attention_mask=mask,
                past_key_values=past_key_values,
                ### TODO **kwargs,
            )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            return hidden_states, past_key_values, shared_kv_states, all_hidden_states
        return hidden_states, past_key_values, shared_kv_states

    def _compute_per_layer_inputs(self,input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if input_ids is None:
            with torch.no_grad():
                mask = inputs_embeds[:, :, None, :] == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5
                input_ids = mask.all(dim=3).nonzero()[:, 2].view(inputs_embeds.shape[:2])
        token_identity = self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape, self.config.num_hidden_layers, self.hidden_size_per_layer_input)
        return token_identity
    
    def _project_per_layer_inputs(
            self, 
            inputs_embeds: torch.Tensor, 
            per_layer_inputs: torch.Tensor,
        ) -> torch.Tensor:
        context_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        context_projection = context_projection.reshape(
            *inputs_embeds.shape[:-1], 
            self.config.num_hidden_layers, 
            self.hidden_size_per_layer_input,
        )
        context_projection = self.per_layer_projection_norm(context_projection)
        combined = (context_projection + per_layer_inputs) * self.per_layer_input_scale
        return combined

# ════════════════════════════════════════════════════════════════════
# CausalLM wrapper
# ════════════════════════════════════════════════════════════════════

class Gemma4ForCausalLM(nn.Module):
    """Gemma-4 text model with LM head for causal language modeling.

    No dependency on HuggingFace transformers (no GenerationMixin, no PreTrainedModel).
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.model = Gemma4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.final_logit_softcapping = config.final_logit_softcapping

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        shared_kv_states: dict | None = None,
        output_hidden_states: bool = False,
    ) -> dict:
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            shared_kv_states=shared_kv_states,
            output_hidden_states=output_hidden_states,
        )
        if output_hidden_states:
            hidden_states, past_key_values, shared_kv_states, all_hidden_states = model_output
        else:
            hidden_states, past_key_values, shared_kv_states = model_output
            all_hidden_states = None
        logits = self.lm_head(hidden_states)

        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": past_key_values,
            "shared_kv_states": shared_kv_states,
            "hidden_states": all_hidden_states if output_hidden_states else None,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float | None = None,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Simplified autoregressive generation (greedy or sampling)."""
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        past_key_values = None
        shared_kv_states = None

        for _ in range(max_new_tokens):
            # On subsequent steps, only feed the last token
            if past_key_values is not None:
                step_ids = input_ids[:, -1:]
            else:
                step_ids = input_ids

            outputs = self(
                input_ids=step_ids,
                past_key_values=past_key_values,
                shared_kv_states=shared_kv_states,
            )

            past_key_values = outputs["past_key_values"]
            shared_kv_states = outputs["shared_kv_states"]
            logits = outputs["logits"][:, -1, :] / temperature

            if do_sample:
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if (next_token == eos_token_id).any():
                break

        return input_ids


# ════════════════════════════════════════════════════════════════════
# Weight loading utilities
# ════════════════════════════════════════════════════════════════════

def load_gemma4_weights(
    model: Gemma4ForCausalLM,
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> Gemma4ForCausalLM:
    """Load Gemma-4 weights from a HuggingFace safetensors checkpoint.

    Handles key remapping from transformers naming conventions to this
    standalone implementation. Supports both single-file and sharded checkpoints.

    Args:
        model: Target Gemma4ForCausalLM instance.
        model_path: Local path or HuggingFace model ID.
        device: Target device for loading.
        dtype: Optional dtype to cast weights to.

    Returns:
        The model with loaded weights.
    """
    import json
    import os

    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors is required for weight loading. Install: pip install safetensors")

    def _find_shards(root: str) -> list[tuple[str, str]]:
        index_path = os.path.join(root, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shards = sorted(set(weight_map.values()))
            return [(s, os.path.join(root, s)) for s in shards]
        single = os.path.join(root, "model.safetensors")
        if os.path.isfile(single):
            return [("model.safetensors", single)]
        return []

    shards = _find_shards(model_path)
    if not shards:
        raise FileNotFoundError(f"No safetensors file found in {model_path}")

    param_dict = dict(model.named_parameters())
    loaded_keys = set()

    for _shard_name, shard_path in shards:
        state = load_file(shard_path, device="cpu")

        # Remap keys: transformers prefix -> our prefix
        remapped = {}
        for k, v in state.items():
            new_k = k
            # Replace "model.language_model." with "model."
            if new_k.startswith("model.language_model."):
                new_k = "model." + new_k[len("model.language_model."):]
            remapped[new_k] = v

        for key, value in remapped.items():
            if key in param_dict:
                if dtype is not None:
                    value = value.to(dtype=dtype)
                param = param_dict[key]
                if param.shape != value.shape:
                    print(f"[WARN] Shape mismatch for {key}: model {param.shape}, checkpoint {value.shape}. Skipping.")
                    continue
                with torch.no_grad():
                    param.copy_(value.to(device=device))
                loaded_keys.add(key)
            elif key == "lm_head.weight":
                embed_key = "embed_tokens.weight"
                if embed_key in param_dict:
                    with torch.no_grad():
                        param_dict[embed_key].copy_(value.to(device=device, dtype=dtype or value.dtype))
                    with torch.no_grad():
                        model.lm_head.weight.data.copy_(param_dict[embed_key].data)
                    loaded_keys.add(embed_key)
                    loaded_keys.add("lm_head.weight")

    if "lm_head.weight" not in loaded_keys and "model.embed_tokens.weight" in loaded_keys:
        with torch.no_grad():
            model.lm_head.weight.data.copy_(param_dict["model.embed_tokens.weight"].data)
        loaded_keys.add("lm_head.weight")

    missing = set(param_dict) - loaded_keys
    if missing:
        print(f"[WARN] {len(missing)} keys not loaded: {sorted(list(missing))[:8]}...")

    return model


__all__ = [
    "Gemma4RMSNorm",
    "Gemma4TextMLP",
    "Gemma4TextAttention",
    "Gemma4TextDecoderLayer",
    "Gemma4TextModel",
    "Gemma4ForCausalLM",
    "Gemma4TextRotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "apply_rotary_pos_emb_qk",
    "repeat_kv",
    "eager_attention_forward",
    "load_gemma4_weights",
]
