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
    sliding_window: int | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager (non-flash) scaled dot-product attention with optional soft-capping."""

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    if attention_mask is not None:
        # Support both additive (float) and boolean masks
        if attention_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask

    # Sliding window: mask out tokens outside the window
    if sliding_window is not None:
        seq_len = attn_weights.shape[-1]
        window_mask = torch.ones(seq_len, seq_len, device=attn_weights.device, dtype=torch.bool).tril()
        window_mask = window_mask.tril(-1) | window_mask.triu(seq_len - sliding_window + 1)
        attn_weights = attn_weights.masked_fill(~window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

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
        self._attention_scalings: dict[str, float] = {}

        for layer_type in self.layer_types:
            rope_params = config.rope_parameters.get(layer_type)
            if rope_params is None:
                continue

            rope_type = rope_params.get("rope_type", "default")
            base = rope_params.get("rope_theta", 10_000.0)
            partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)

            if layer_type == "full_attention" and rope_type == "proportional":
                dim = config.global_head_dim
            else:
                dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

            dim = int(dim * partial_rotary_factor)

            attention_factor = 1.0
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
            )
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self._attention_scalings[layer_type] = attention_factor

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str = "full_attention"):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attn_scaling = self._attention_scalings[layer_type]

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
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings

        # --- Q projection ---
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # --- KV projection (or reuse shared) ---
        if self.is_kv_shared_layer and shared_kv_states is not None:
            key_states, value_states = shared_kv_states[self.layer_type]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            if self.v_proj is not None:
                value_states = self.v_proj(hidden_states).view(hidden_shape)
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
            sliding_window=self.sliding_window,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        shared_kv_states: dict | None = None,
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

        return hidden_states


class Gemma4TextModel(nn.Module):
    """Pure PyTorch implementation of the Gemma-4 Text Model (decoder-only)."""

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([
            Gemma4TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)

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
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        _, seq_len = inputs_embeds.shape[:2]

        # Convert 2D mask to 4D attention mask
        if attention_mask.dim() == 2:
            expanded_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
            expanded_mask = (1.0 - expanded_mask) * torch.finfo(inputs_embeds.dtype).min
            return expanded_mask
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

        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            pos_emb = rope_embs[layer_type]

            mask = causal_mask
            if layer_type == "sliding_attention":
                mask = causal_mask  # sliding window is handled inside eager_attention_forward

            hidden_states = layer(
                hidden_states,
                position_embeddings=pos_emb,
                attention_mask=mask,
                past_key_values=past_key_values,
                shared_kv_states=shared_kv_states,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values, shared_kv_states


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
    ) -> dict:
        hidden_states, past_key_values, shared_kv_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            shared_kv_states=shared_kv_states,
        )
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
            # Strip "model." prefix
            if new_k.startswith("model."):
                new_k = new_k[6:]
            remapped[new_k] = v

        for key, value in remapped.items():
            if key in param_dict:
                if dtype is not None:
                    value = value.to(dtype=dtype)
                param = param_dict[key]
                with torch.no_grad():
                    param.copy_(value.to(device=device))
                loaded_keys.add(key)
            elif key == "lm_head.weight":
                # tie_word_embeddings: share with embed_tokens
                embed_key = "embed_tokens.weight"
                if embed_key in param_dict:
                    with torch.no_grad():
                        param_dict[embed_key].copy_(value.to(device=device, dtype=dtype or value.dtype))
                    loaded_keys.add(embed_key)

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
