#!/usr/bin/env python3
"""
Qwen3.5 Pure PyTorch Inference Script

Minimal implementation of Qwen3.5 hybrid architecture
(Gated DeltaNet + Gated Attention + MoE) using only PyTorch.

Supports:
  - Model definition (full architecture)
  - Weight loading from HuggingFace safetensors (built-in parser, no extra deps)
  - Text generation

Usage:
  python qwen3_5_pure_torch.py --model-path /path/to/Qwen3.5-397B-A17B --prompt "Hello"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import struct
import os
import sys
import math
import time
from typing import Optional, List, Dict
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer


# ============================================================
# Configuration
# ============================================================

@dataclass
class Qwen3_5Config:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 60
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 64
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_dim: int = 1024
    full_attention_interval: int = 4
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    @classmethod
    def from_pretrained(cls, model_path_or_id: str, **kwargs):
        hf_config = AutoConfig.from_pretrained(model_path_or_id, **kwargs)
        return cls.from_hf_config(hf_config.to_dict())

    @classmethod
    def from_hf_config(cls, config_dict: dict):
        cfg = config_dict.get("text_config", config_dict)
        return cls(
            hidden_size=cfg.get("hidden_size", 4096),
            intermediate_size=cfg.get("intermediate_size", 12288),
            num_hidden_layers=cfg.get("num_hidden_layers", 60),
            num_attention_heads=cfg.get("num_attention_heads", 32),
            num_key_value_heads=cfg.get("num_key_value_heads", 2),
            head_dim=cfg.get("head_dim", 256),
            vocab_size=cfg.get("vocab_size", 248320),
            max_position_embeddings=cfg.get("max_position_embeddings", 262144),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=cfg.get("rope_theta", 10000000.0),
            partial_rotary_factor=cfg.get("partial_rotary_factor", 0.25),
            linear_num_key_heads=cfg.get("linear_num_key_heads", 16),
            linear_num_value_heads=cfg.get("linear_num_value_heads", 64),
            linear_key_head_dim=cfg.get("linear_key_head_dim", 128),
            linear_value_head_dim=cfg.get("linear_value_head_dim", 128),
            linear_conv_kernel_dim=cfg.get("linear_conv_kernel_dim", 4),
            num_experts=cfg.get("num_experts", 0),
            num_experts_per_tok=cfg.get("num_experts_per_tok", 0),
            moe_intermediate_dim=cfg.get("moe_intermediate_dim", cfg.get("intermediate_size", 1024)),
            full_attention_interval=cfg.get("full_attention_interval", 4),
            attention_bias=cfg.get("attention_bias", False),
            attention_dropout=cfg.get("attention_dropout", 0.0),
            hidden_act=cfg.get("hidden_act", "silu"),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )


# ============================================================
# Utility Modules
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(gate.dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(gate.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000000.0, max_seq_len: int = 262144):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        inv_freq = self.inv_freq[None, :, None].float()
        position_ids = position_ids[:, None, :].float()
        freqs = inv_freq @ position_ids
        freqs = freqs.transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.to(x.dtype)

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        cos = cos[:x.shape[-2], :x.shape[-1]]
        sin = sin[:x.shape[-2], :x.shape[-1]]
        x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated


# ============================================================
# Gated DeltaNet (Linear Attention)
# ============================================================

class ShortConvolution(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              groups=channels, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch_shape, seq_len, channels = x.shape
        x = x.reshape(-1, seq_len, channels)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        x = F.silu(x)
        x = x.permute(0, 2, 1)
        batch = batch_shape[0] if batch_shape else 1
        x = x.reshape(*batch_shape, seq_len, channels)
        return x


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.kv_ratio = self.num_v_heads // self.num_k_heads
        self.conv_kernel = config.linear_conv_kernel_dim

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.conv1d = ShortConvolution(
            self.key_dim * 2 + self.value_dim, self.conv_kernel
        )

        self.A_log = nn.Parameter(torch.empty(1, 1, self.num_v_heads))
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads))

        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.layer_idx = layer_idx

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.A_log, a=math.log(1e-4), b=math.log(1e-2))
        nn.init.ones_(self.dt_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple:
        batch, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        q = mixed_qkv[..., :self.key_dim]
        k = mixed_qkv[..., self.key_dim:self.key_dim * 2]
        v = mixed_qkv[..., self.key_dim * 2:]

        mixed_qkv = torch.cat([q, k, v], dim=-1)
        mixed_qkv = self.conv1d(mixed_qkv)

        q = mixed_qkv[..., :self.key_dim]
        k = mixed_qkv[..., self.key_dim:self.key_dim * 2]
        v = mixed_qkv[..., self.key_dim * 2:self.key_dim * 2 + self.value_dim]

        q = q.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.view(batch, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.view(batch, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.view(batch, seq_len, self.num_v_heads, self.head_v_dim)

        q = q.repeat_interleave(self.kv_ratio, dim=2)
        k = k.repeat_interleave(self.kv_ratio, dim=2)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        beta = torch.sigmoid(b.to(torch.float32)).to(hidden_states.dtype)
        g = -self.A_log.exp() * F.softplus(a.to(torch.float32) + self.dt_bias)
        g = g.to(hidden_states.dtype)

        output, new_state = self._gated_delta_rule(q, k, v, g, beta, past_state)

        output = self.norm(output.view(-1, self.head_v_dim),
                           z.reshape(-1, self.head_v_dim))
        output = output.view(batch, seq_len, self.value_dim)
        output = self.out_proj(output)

        return output, new_state

    def _gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_log: torch.Tensor,
        beta: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple:
        batch, seq, num_heads, d_k = q.shape
        d_v = v.shape[-1]

        if state is None:
            state = torch.zeros(batch, num_heads, d_k, d_v,
                                device=q.device, dtype=q.dtype)

        outputs = []
        for t in range(seq):
            qt = q[:, t, :, :]
            kt = k[:, t, :, :]
            vt = v[:, t, :, :]
            gt = gate_log[:, t, :]
            bt = beta[:, t, :]

            gate = torch.exp(gt)
            state = state * gate.unsqueeze(-1).unsqueeze(-1)

            kv_mem = torch.einsum('bhkv,bhk->bhv', state, kt)
            delta = bt.unsqueeze(-1) * (vt - kv_mem)

            state = state + torch.einsum('bhk,bhv->bhkv', kt, delta)

            out = torch.einsum('bhkv,bhk->bhv', state, qt)
            outputs.append(out)

        out = torch.stack(outputs, dim=2).permute(0, 2, 1, 3)
        return out, state


# ============================================================
# Standard Gated Attention (Softmax Attention)
# ============================================================

class GatedAttention(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)

        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.q_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=config.attention_bias)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary = RotaryEmbedding(
            self.rotary_dim, theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings
        )
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_kv: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tuple:
        batch, seq_len, _ = hidden_states.shape

        q_gate = self.q_proj(hidden_states)
        q, gate = q_gate.chunk(2, dim=-1)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        gate = gate.view(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = None, None
        if self.rotary_dim > 0:
            cos, sin = self._get_cos_sin(position_ids, q.device, q.dtype)
            q_rot, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
            k_rot, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]
            q_rot = RotaryEmbedding.apply_rotary(q_rot, cos, sin)
            k_rot = RotaryEmbedding.apply_rotary(k_rot, cos, sin)
            q = torch.cat([q_rot, q_pass], dim=-1)
            k = torch.cat([k_rot, k_pass], dim=-1)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_k = k
        current_v = v

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.full((seq_len, k.size(-2)), float('-inf'),
                       device=q.device, dtype=q.dtype), diagonal=1
        )[-seq_len:, :]
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        attn_output = attn_output * torch.sigmoid(gate.reshape(batch, seq_len, -1))
        attn_output = self.o_proj(attn_output)

        new_kv = (current_k, current_v) if use_cache else None
        return attn_output, new_kv

    def _get_cos_sin(self, position_ids, device, dtype):
        inv_freq = 1.0 / (self.rotary.theta ** (
            torch.arange(0, self.rotary_dim, 2, device=device, dtype=torch.float32) / self.rotary_dim
        ))
        inv_freq = inv_freq[None, :, None]
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# ============================================================
# Mixture of Experts
# ============================================================

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class MoEExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.intermediate_dim = config.moe_intermediate_dim

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.shared_expert = MoEExpert(self.hidden_size, self.intermediate_dim)
        self.experts = nn.ModuleList([
            MoEExpert(self.hidden_size, self.intermediate_dim)
            for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = hidden_states.shape
        x = hidden_states.view(-1, hidden_size)

        router_logits = self.router(x)
        router_weights = F.softmax(router_logits.float(), dim=-1).to(x.dtype)

        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)

        shared_out = self.shared_expert(x)

        routed_out = torch.zeros_like(x)
        for i in range(self.top_k):
            weights = top_k_weights[:, i]
            indices = top_k_indices[:, i]
            for expert_idx in range(self.num_experts):
                mask = indices == expert_idx
                if mask.any():
                    routed_out[mask] += weights[mask, None] * self.experts[expert_idx](x[mask])

        output = shared_out + routed_out
        return output.view(batch, seq_len, hidden_size)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full_attn = (
            config.num_experts > 0 and
            (layer_idx % config.full_attention_interval) == (config.full_attention_interval - 1)
        ) or (
            config.num_experts == 0 and layer_idx >= config.num_hidden_layers // 2
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.is_full_attn:
            self.self_attn = GatedAttention(config)
        else:
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.num_experts > 0 and config.num_experts_per_tok > 0:
            self.mlp = MoE(config)
        else:
            self.mlp = DenseMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_state: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple] = None,
        use_cache: bool = False,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.is_full_attn:
            attn_out, new_kv = self.self_attn(
                hidden_states, position_ids, past_kv, use_cache
            )
            new_state = None
        else:
            attn_out, new_state = self.linear_attn(
                hidden_states, position_ids, past_state, use_cache
            )
            new_kv = None

        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = residual + mlp_out

        return hidden_states, new_state, new_kv


class DenseMLP(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# Full Model
# ============================================================

class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_states: Optional[List[torch.Tensor]] = None,
        past_kvs: Optional[List[tuple]] = None,
        use_cache: bool = False,
    ) -> tuple:
        batch, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)

        new_states = [] if use_cache else None
        new_kvs = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_state = past_states[i] if past_states else None
            past_kv = past_kvs[i] if past_kvs else None

            hidden_states, new_state, new_kv = layer(
                hidden_states, position_ids, past_state, past_kv, use_cache
            )
            if use_cache:
                if new_state is not None:
                    new_states.append(new_state)
                if new_kv is not None:
                    new_kvs.append(new_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, new_states, new_kvs


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_states: Optional[List[torch.Tensor]] = None,
        past_kvs: Optional[List[tuple]] = None,
        use_cache: bool = False,
    ) -> tuple:
        hidden_states, new_states, new_kvs = self.model(
            input_ids, position_ids, past_states, past_kvs, use_cache
        )
        logits = self.lm_head(hidden_states)
        return logits, new_states, new_kvs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.95,
        eos_token_id: int = 248044,
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        batch, prompt_len = input_ids.shape

        past_states = [None] * self.config.num_hidden_layers
        past_kvs = [None] * self.config.num_hidden_layers

        position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
        logits, past_states, past_kvs = self(
            input_ids, position_ids, past_states, past_kvs, use_cache=True
        )

        next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)
        generated = [next_token.item()]

        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break

            position_ids = torch.tensor([[prompt_len + len(generated) - 1]], device=device)
            logits, past_states, past_kvs = self(
                next_token, position_ids, past_states, past_kvs, use_cache=True
            )
            next_token = self._sample(logits[:, -1, :], temperature, top_k, top_p)
            generated.append(next_token.item())
            if next_token.item() == eos_token_id:
                break

        return torch.cat([input_ids, torch.tensor([generated], device=device)], dim=-1)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        logits = logits / max(temperature, 1e-5)

        if top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, -1:]] = float('-inf')

        if top_p < 1.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


# ============================================================
# Minimal Safetensors Loader (pure Python, no extra deps)
# ============================================================

def load_safetensors_file(path: str) -> Dict[str, torch.Tensor]:
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode('utf-8'))

        result = {}
        for key, metadata in header.items():
            if key == '__metadata__':
                continue
            offset = metadata['data_offsets'][0]
            size = metadata['data_offsets'][1] - offset
            f.seek(8 + header_len + offset)
            tensor_bytes = f.read(size)
            dtype = _safetensors_dtype_to_torch(metadata['dtype'])
            shape = metadata['shape']
            tensor = torch.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)
            result[key] = tensor
    return result


def _safetensors_dtype_to_torch(dtype_str: str) -> torch.dtype:
    mapping = {
        'F16': torch.float16,
        'F32': torch.float32,
        'BF16': torch.bfloat16,
        'I8': torch.int8,
        'I16': torch.int16,
        'I32': torch.int32,
        'I64': torch.int64,
        'U8': torch.uint8,
    }
    return mapping.get(dtype_str, torch.float32)


def find_safetensors_files(model_path: str) -> List[str]:
    files = []
    for f in sorted(os.listdir(model_path)):
        if f.endswith('.safetensors'):
            files.append(os.path.join(model_path, f))
    return files


def resolve_model_path(model_path_or_id: str, token: Optional[str] = None) -> str:
    if os.path.exists(model_path_or_id):
        return os.path.abspath(model_path_or_id)

    print("=" * 60)
    print(f" Model '{model_path_or_id}' not found locally.")
    print()
    print(" Download it from HuggingFace Hub first:")
    print(f"   pip install huggingface_hub")
    print(f"   huggingface-cli download {model_path_or_id} \\")
    print(f"     --local-dir ./{model_path_or_id.replace('/', '_')}")
    print()
    print(" Then re-run with:")
    print(f"   --model-path ./{model_path_or_id.replace('/', '_')}")
    print("=" * 60)
    raise FileNotFoundError(f"Model not found: {model_path_or_id}")


# ============================================================
# Weight Loading
# ============================================================

def load_qwen3_5_weights(model: Qwen3_5ForCausalLM, model_path: str, dtype=torch.bfloat16):
    safetensors_files = find_safetensors_files(model_path)
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        model.config = Qwen3_5Config.from_hf_config(cfg)

    print(f"Loading {len(safetensors_files)} safetensors file(s)...")
    state_dict = {}
    for sf in safetensors_files:
        print(f"  Loading {os.path.basename(sf)}...")
        tensors = load_safetensors_file(sf)
        state_dict.update(tensors)

    print(f"\nAvailable HF keys: {len(state_dict)}")
    hf_sample = sorted(state_dict.keys())[:5]
    for k in hf_sample:
        print(f"  e.g. {k} shape={state_dict[k].shape}")

    model_state = model.state_dict()
    needed = set(model_state.keys())
    available = set(state_dict.keys())

    print(f"\nModel keys needed: {len(needed)}")
    model_sample = sorted(needed)[:5]
    for k in model_sample:
        print(f"  e.g. {k} shape={model_state[k].shape}")

    mapping = _build_weight_map(model, state_dict)

    loaded = 0
    missing = []
    for name, param in model.named_parameters():
        candidates = [name]
        if name in mapping:
            candidates.insert(0, mapping[name])

        found = False
        for hf_name in candidates:
            if hf_name in state_dict:
                w = state_dict[hf_name].to(dtype)
                if w.shape == param.shape:
                    param.data.copy_(w)
                    loaded += 1
                    found = True
                    break
                else:
                    print(f"  Shape mismatch: {name} {list(param.shape)} vs {hf_name} {list(w.shape)}")
                    if hf_name != name:
                        if name in state_dict:
                            w2 = state_dict[name].to(dtype)
                            if w2.shape == param.shape:
                                param.data.copy_(w2)
                                loaded += 1
                                found = True
                                print(f"    (matched using direct name instead)")
                                break

        if not found and name not in (
            'A_log', 'dt_bias',
            *(f'{k}.A_log' for k in _attn_names(model)),
            *(f'{k}.dt_bias' for k in _attn_names(model)),
        ):
            missing.append(name)

    total = sum(1 for _ in model.named_parameters())
    print(f"Loaded {loaded} / {total} weights")
    if missing:
        print(f"  Missing {len(missing)} weights (first 10):")
        for n in missing[:10]:
            print(f"    - {n}")
    return model


def _attn_names(model):
    names = []
    for n, _ in model.named_modules():
        parts = n.split('.')
        for i, p in enumerate(parts):
            if p in ('self_attn', 'linear_attn'):
                names.append('.'.join(parts[:i+1]))
    return list(set(names))


def _build_weight_map(model, state_dict):
    """
    Build a mapping from our model's parameter names to HF checkpoint names.
    Handles the differences in naming conventions:
      - Our 'linear_attn.*' -> HF 'model.layers.N.linear_attn.*'
      - Our 'self_attn.*'   -> HF 'model.layers.N.self_attn.*'
      - Our 'mlp.*'        -> HF 'model.layers.N.mlp.*' (dense or MoE)
    """
    hf_prefix = 'model.layers.'
    mapping = {}

    for name, _ in model.named_parameters():
        if not name.startswith('model.layers.'):
            continue

        rest = name[len('model.layers.'):]
        dot = rest.index('.') if '.' in rest else -1
        if dot < 0:
            continue
        layer_idx = rest[:dot]
        layer_rest = rest[dot+1:]

        if layer_rest.startswith('linear_attn'):
            sub = layer_rest[len('linear_attn.'):]
            if sub.startswith('in_proj_qkv'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.in_proj_qkv.weight'
            elif sub.startswith('in_proj_z'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.in_proj_z.weight'
            elif sub.startswith('in_proj_b'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.in_proj_b.weight'
            elif sub.startswith('in_proj_a'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.in_proj_a.weight'
            elif sub.startswith('conv1d.conv'):
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.conv1d.weight'
            else:
                mapping[name] = f'{hf_prefix}{layer_idx}.linear_attn.{sub}'
        elif layer_rest.startswith('self_attn'):
            mapping[name] = f'{hf_prefix}{layer_idx}.self_attn.{layer_rest[len("self_attn."):]}'
        elif layer_rest.startswith('input_layernorm') or layer_rest.startswith('post_attention_layernorm'):
            mapping[name] = f'{hf_prefix}{layer_idx}.{layer_rest}'
        elif layer_rest.startswith('mlp'):
            mlp_rest = layer_rest[len('mlp.'):]
            if mlp_rest.startswith('router'):
                mapping[name] = f'{hf_prefix}{layer_idx}.mlp.gate.{mlp_rest[len("router."):]}'
            elif mlp_rest.startswith('shared_expert'):
                mapping[name] = f'{hf_prefix}{layer_idx}.mlp.shared_expert.{mlp_rest[len("shared_expert."):]}'
            elif mlp_rest.startswith('experts'):
                mapping[name] = f'{hf_prefix}{layer_idx}.mlp.{mlp_rest}'
            elif mlp_rest in ('gate_proj.weight', 'up_proj.weight', 'down_proj.weight'):
                mapping[name] = f'{hf_prefix}{layer_idx}.mlp.{mlp_rest}'
            else:
                mapping[name] = f'{hf_prefix}{layer_idx}.{layer_rest}'
        else:
            mapping[name] = f'{hf_prefix}{layer_idx}.{layer_rest}'

    non_layer_keys = {
        'model.embed_tokens.weight': 'model.embed_tokens.weight',
        'model.norm.weight': 'model.norm.weight',
        'lm_head.weight': 'lm_head.weight',
    }
    for our_key, hf_key in non_layer_keys.items():
        mapping[our_key] = hf_key

    return mapping


# ============================================================
# Tokenizer (using transformers)
# ============================================================

def build_tokenizer(model_path_or_id: str, **kwargs) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Qwen3.5 Pure PyTorch Inference')
    parser.add_argument('--model-path', type=str, required=True,
                        help='HF model ID (e.g. Qwen/Qwen3.5-397B-A17B) or local path')
    parser.add_argument('--prompt', type=str, default='Hello, how are you?',
                        help='Input prompt')
    parser.add_argument('--max-new-tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Top-k sampling parameter')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model dtype')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (cuda, cpu, auto)')
    parser.add_argument('--just-print-config', action='store_true',
                        help='Print model config and exit')

    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map[args.dtype]

    try:
        model_path = resolve_model_path(args.model_path)
    except FileNotFoundError as e:
        print(e)
        return

    config = Qwen3_5Config.from_pretrained(model_path)

    print(f"\nQwen3.5 Configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  KV heads: {config.num_key_value_heads}")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Linear attention: {config.linear_num_key_heads} K heads, {config.linear_num_value_heads} V heads")
    print(f"  Experts: {config.num_experts}, top-{config.num_experts_per_tok}")
    print(f"  Max pos embeddings: {config.max_position_embeddings}")
    print(f"  Device: {device}")
    print(f"  Dtype: {args.dtype}")

    if args.just_print_config:
        return

    print("\nBuilding model...")
    t0 = time.time()
    model = Qwen3_5ForCausalLM(config)
    model = model.to(dtype).to(device)
    print(f"  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  Active parameters: ~{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\nLoading weights...")
    load_qwen3_5_weights(model, args.model_path, dtype)
    model = model.to(device)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("\nPreparing tokenizer...")
    tokenizer = build_tokenizer(args.model_path)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    print(f"\nPrompt: {args.prompt}")
    print(f"Input tokens: {input_ids.shape[-1]}")

    with torch.no_grad():
        if device == 'cuda':
            torch.cuda.synchronize()

        t1 = time.time()
        generated = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
        if device == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()

    output_text = tokenizer.decode(generated[0].tolist())
    print(f"\nGenerated ({generated.shape[-1] - input_ids.shape[-1]} tokens in {t2-t1:.2f}s):")
    print(output_text)


if __name__ == '__main__':
    main()
