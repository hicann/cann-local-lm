"""Qwen3 (standard) pure PyTorch implementation - no transformers dependency.

Based on HuggingFace transformers/src/transformers/models/qwen3/modeling_qwen3.py.
Standard Qwen3 is the base architecture of Qwen3.5: a plain decoder-only
transformer with GQA, QK-norm, SwiGLU (SiLU) MLP and default full-rotary RoPE.
It does NOT carry the Q Gate / GatedDeltaNet linear attention / MRoPE /
centered RMSNorm that Qwen3.5 adds on top.

Implemented following the qwen3-5-pure-torch SKILL.md conventions.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .configuration_qwen3 import Qwen3TextConfig


# ════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    cos/sin are (batch, seq, rotary_dim) and are unsqueezed at `unsqueeze_dim`
    so q/k of shape (batch, heads, seq, head_dim) broadcast correctly.
    Supports partial rotary via rotary_dim = cos.shape[-1].
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager (non-flash) scaled dot-product attention."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ════════════════════════════════════════════════════════════════════
# RMSNorm (standard: weight init ones, output = weight * normalized)
# ════════════════════════════════════════════════════════════════════

class Qwen3RMSNorm(nn.Module):
    """Standard RMSNorm matching the HuggingFace Qwen3 convention.

    Unlike Qwen3.5's centered RMSNorm (1 + weight), Qwen3 uses the plain
    `weight * normalized` form with weight initialised to ones.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# ════════════════════════════════════════════════════════════════════
# Rotary Embeddings (default, full-rotary)
# ════════════════════════════════════════════════════════════════════

class Qwen3RotaryEmbedding(nn.Module):
    """Default full-rotary RoPE for standard Qwen3.

    No MRoPE / partial rotary like Qwen3.5; rotates the full head_dim.
    """

    def __init__(self, config: Qwen3TextConfig, device=None):
        super().__init__()
        base = config.rope_theta
        dim = config.head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(x.device, dtype=x.dtype)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ════════════════════════════════════════════════════════════════════
# MLP (SwiGLU with SiLU)
# ════════════════════════════════════════════════════════════════════

class Qwen3MLP(nn.Module):
    """Gated MLP (SwiGLU) for Qwen3 decoder."""

    def __init__(self, config: Qwen3TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ════════════════════════════════════════════════════════════════════
# Attention (GQA + QK-norm, NO Q Gate)
# ════════════════════════════════════════════════════════════════════

class Qwen3Attention(nn.Module):
    """Multi-headed attention for Qwen3 decoder.

    Standard Qwen3 attention: GQA with separate q/k/v/o projections and
    per-head QK-norm (RMSNorm over head_dim). No Q Gate (that is Qwen3.5-only).
    """

    def __init__(self, config: Qwen3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        new_kv_cache_pos: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states).view(*input_shape, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache = past_key_values[self.layer_idx]
            if new_kv_cache_pos is not None and cache is not None:
                # scatter-write: index_copy 到预分配 buffer (K=kv_max 固定)
                cache_k = cache[0].index_copy(2, new_kv_cache_pos, key_states)
                cache_v = cache[1].index_copy(2, new_kv_cache_pos, value_states)
                past_key_values[self.layer_idx] = (cache_k, cache_v)
                key_states = cache_k
                value_states = cache_v
            elif cache is not None:
                # fallback: torch.cat (K=kv_max+seq_len, 非 scatter, eager generate 用)
                key_states = torch.cat([cache[0], key_states], dim=2)
                value_states = torch.cat([cache[1], value_states], dim=2)
                past_key_values[self.layer_idx] = (key_states, value_states)
            else:
                past_key_values[self.layer_idx] = (key_states, value_states)

        attn_output, _ = eager_attention_forward(
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
# Decoder Layer
# ════════════════════════════════════════════════════════════════════

class Qwen3DecoderLayer(nn.Module):
    """Single decoder layer: pre-norm attention + pre-norm MLP.

    Norm layout is the standard Qwen layout (only input_layernorm and
    post_attention_layernorm), matching the HF safetensors keys.
    """

    def __init__(self, config: Qwen3TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: list | None = None,
        new_kv_cache_pos: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            new_kv_cache_pos=new_kv_cache_pos,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ════════════════════════════════════════════════════════════════════
# Text Model
# ════════════════════════════════════════════════════════════════════

class Qwen3TextModel(nn.Module):
    """Pure PyTorch implementation of the Qwen3 Text Model (decoder-only)."""

    def __init__(self, config: Qwen3TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def _prepare_position_ids(self, inputs_embeds: torch.Tensor, past_key_values: list | None) -> torch.Tensor:
        batch_size, seq_len = inputs_embeds.shape[:2]
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[2]
        device = inputs_embeds.device
        return torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0)

    def _prepare_causal_mask(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: list | None,
    ) -> torch.Tensor | None:
        bsz, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Single-token decode step with an active KV cache: the single query
        # may attend to all cached keys, no additive mask is needed.
        # (scatter 模式传 4D mask 屏蔽未写入位置, 不 return None)
        if (past_key_values is not None and past_key_values[0] is not None
                and seq_len == 1 and attention_mask is None):
            return None

        if attention_mask is None:
            min_dtype = torch.finfo(dtype).min
            past_len = past_key_values[0][0].shape[2] if (
                past_key_values is not None and past_key_values[0] is not None) else 0
            total_len = past_len + seq_len
            q_pos = torch.arange(seq_len, device=device).unsqueeze(1) + past_len
            k_pos = torch.arange(total_len, device=device).unsqueeze(0)
            mask = torch.where(k_pos <= q_pos, torch.tensor(0.0, dtype=dtype, device=device), min_dtype)
            return mask.unsqueeze(0).unsqueeze(0)

        if attention_mask.dim() == 2:
            key_valid = attention_mask.to(torch.bool)[:, None, :]
            causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            full_mask = key_valid.expand(bsz, seq_len, seq_len) & causal
            min_dtype = torch.finfo(dtype).min
            return torch.where(full_mask, 0.0, min_dtype).to(dtype).unsqueeze(0).unsqueeze(0)

        return attention_mask

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        new_kv_cache_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list | None]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = self._prepare_position_ids(inputs_embeds, past_key_values)

        causal_mask = self._prepare_causal_mask(inputs_embeds, attention_mask, past_key_values)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                new_kv_cache_pos=new_kv_cache_pos,
            )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            return hidden_states, past_key_values, all_hidden_states
        return hidden_states, past_key_values


# ════════════════════════════════════════════════════════════════════
# CausalLM wrapper
# ════════════════════════════════════════════════════════════════════

class Qwen3ForCausalLM(nn.Module):
    """Qwen3 text model with LM head for causal language modeling.

    No dependency on HuggingFace transformers (no GenerationMixin, no PreTrainedModel).
    """

    def __init__(self, config: Qwen3TextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: list | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        new_kv_cache_pos: torch.Tensor | None = None,
    ) -> dict:
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            new_kv_cache_pos=new_kv_cache_pos,
        )
        if output_hidden_states:
            hidden_states, past_key_values, all_hidden_states = model_output
        else:
            hidden_states, past_key_values = model_output
            all_hidden_states = None

        logits = self.lm_head(hidden_states)

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
        eos_token_id: int | list[int] | None = None,
    ) -> torch.Tensor:
        """Simplified autoregressive generation (greedy or sampling)."""
        eos_ids = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        if isinstance(eos_ids, int):
            eos_set = {eos_ids}
        elif isinstance(eos_ids, (list, tuple)):
            eos_set = set(eos_ids)
        else:
            eos_set = set()

        past_key_values = None
        num_layers = self.config.num_hidden_layers
        for _ in range(max_new_tokens):
            if past_key_values is None:
                # First step: prefill the full prompt and initialise the KV cache
                # (forward only builds/uses the cache when a list is passed).
                step_ids = input_ids
                past_key_values = [None] * num_layers
            else:
                step_ids = input_ids[:, -1:]
            outputs = self(input_ids=step_ids, past_key_values=past_key_values)
            past_key_values = outputs["past_key_values"]
            logits = outputs["logits"][:, -1, :] / max(temperature, 1e-8)

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
            if eos_set and (next_token == torch.tensor(list(eos_set), device=next_token.device)).any():
                break

        return input_ids


# ════════════════════════════════════════════════════════════════════
# Weight loading utilities
# ════════════════════════════════════════════════════════════════════

def load_qwen3_weights(
    model: Qwen3ForCausalLM,
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> Qwen3ForCausalLM:
    """Load Qwen3 weights from a HuggingFace safetensors checkpoint.

    The HF safetensors key naming (model.layers.N.*, model.embed_tokens.weight,
    model.norm.weight, lm_head.weight, *.q_norm/k_norm.weight, ...) already
    matches this implementation's parameter names, so no remapping is needed.
    Supports both single-file and sharded checkpoints, and HF model IDs.
    """
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors is required for weight loading. Install: pip install safetensors")

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif os.path.isfile(single_path):
        shards = ["model.safetensors"]
    else:
        from huggingface_hub import snapshot_download
        cached = snapshot_download(model_path, allow_patterns=["*.safetensors", "*.json"])
        return load_qwen3_weights(model, cached, device, dtype)

    param_dict = dict(model.named_parameters())
    loaded: set[str] = set()

    for shard in shards:
        shard_path = os.path.join(model_path, shard)
        if not os.path.isfile(shard_path):
            from huggingface_hub import hf_hub_download
            shard_path = hf_hub_download(model_path, shard)
        state = load_file(shard_path, device="cpu")
        filtered = {k: v for k, v in state.items() if k in param_dict}
        cast_target = {}
        for k, v in filtered.items():
            param = param_dict[k]
            if param.shape != v.shape:
                print(f"[WARN] Shape mismatch for {k}: model {tuple(param.shape)}, checkpoint {tuple(v.shape)}. Skipping.")
                continue
            cast_target[k] = v.to(device=device, dtype=dtype) if dtype is not None else v.to(device=device)
        model.load_state_dict(cast_target, strict=False)
        loaded.update(cast_target.keys())

    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    missing = set(param_dict) - loaded
    if missing:
        print(f"[WARN] {len(missing)} keys not loaded: {sorted(list(missing))[:8]}...")

    return model


__all__ = [
    "Qwen3RMSNorm",
    "Qwen3MLP",
    "Qwen3Attention",
    "Qwen3DecoderLayer",
    "Qwen3TextModel",
    "Qwen3ForCausalLM",
    "Qwen3RotaryEmbedding",
    "apply_rotary_pos_emb",
    "repeat_kv",
    "rotate_half",
    "eager_attention_forward",
    "load_qwen3_weights",
]
