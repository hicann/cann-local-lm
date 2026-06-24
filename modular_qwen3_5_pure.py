"""
Qwen3.5 Pure PyTorch — modular building blocks.
No transformers dependency — only torch.

Contains: config, norms, RoPE, delta rule kernels, GatedDeltaNet, Attention, MLP, DecoderLayer, Cache.
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

@dataclass
class Qwen3_5TextConfig:
    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 256
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    attention_bias: bool = False
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    pad_token_id: int = 248044
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 64
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    layer_types: Optional[Tuple[str, ...]] = None
    mrope_section: Tuple[int, ...] = (11, 11, 10)

    def __post_init__(self):
        if self.layer_types is None:
            full_interval = 4
            self.layer_types = tuple(
                "full_attention" if (i % full_interval) == (full_interval - 1) else "linear_attention"
                for i in range(self.num_hidden_layers)
            )

    @classmethod
    def from_dict(cls, d: dict):
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# ============================================================
# Normalization
# ============================================================

class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ============================================================
# RoPE with M-RoPE support
# ============================================================

class Qwen3_5TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        base = config.rope_theta
        partial_rotary_factor = config.partial_rotary_factor
        head_dim = config.head_dim
        dim = int(head_dim * partial_rotary_factor)
        self.dim = dim
        self.mrope_section = config.mrope_section

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def apply_interleaved_mrope(freqs, mrope_section):
        freqs_t = freqs[0]
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim_idx, ..., idx]
        return freqs_t


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# ============================================================
# Causal Conv1D (pure torch fallback)
# ============================================================

def torch_causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


# ============================================================
# Gated Delta Rule implementations (pure torch)
# ============================================================

def l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64,
                                 initial_state=None, output_final_state=False,
                                 use_qk_l2norm_in_kernel=False, **kwargs):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state,
                                     output_final_state, use_qk_l2norm_in_kernel=False):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim,
                                dtype=value.dtype, device=value.device)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ============================================================
# Gated DeltaNet (Linear Attention)
# ============================================================

class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.layer_norm_epsilon = config.rms_norm_eps

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx]["conv_states"]
            recurrent_state = cache_params.layers[self.layer_idx]["recurrent_states"]

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1:
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, "silu"
            )
        else:
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        kv_ratio = self.num_v_heads // self.num_k_heads
        if kv_ratio > 1:
            query = query.repeat_interleave(kv_ratio, dim=2)
            key = key.repeat_interleave(kv_ratio, dim=2)

        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)


# ============================================================
# Standard Gated Attention (Softmax Attention)
# ============================================================

def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class Qwen3_5Attention(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim * 2,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size,
                                bias=config.attention_bias)

        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states_rep = repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_rep.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_rep)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output


# ============================================================
# MLP (SwiGLU)
# ============================================================

class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# Decoder Layer
# ============================================================

class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states, cache_params=past_key_values,
                attention_mask=attention_mask, **kwargs,
            )
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states, attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values, **kwargs,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================
# Cache
# ============================================================

class DynamicCache:
    def __init__(self, config=None):
        self.layers = []
        self._seen_tokens = 0

    def has_previous_state(self, layer_idx):
        if layer_idx >= len(self.layers):
            return False
        return True

    def update(self, key_states, value_states, layer_idx):
        while len(self.layers) <= layer_idx:
            self.layers.append({"key_states": None, "value_states": None})
        if self.layers[layer_idx]["key_states"] is not None:
            key_states = torch.cat([self.layers[layer_idx]["key_states"], key_states], dim=2)
            value_states = torch.cat([self.layers[layer_idx]["value_states"], value_states], dim=2)
        self.layers[layer_idx]["key_states"] = key_states
        self.layers[layer_idx]["value_states"] = value_states
        self._seen_tokens = key_states.shape[2]
        return key_states, value_states

    def update_conv_state(self, state, layer_idx):
        while len(self.layers) <= layer_idx:
            self.layers.append({})
        self.layers[layer_idx]["conv_states"] = state

    def update_recurrent_state(self, state, layer_idx):
        while len(self.layers) <= layer_idx:
            self.layers.append({})
        self.layers[layer_idx]["recurrent_states"] = state

    def get_seq_length(self):
        return self._seen_tokens

    def reset(self):
        self.layers.clear()
        self._seen_tokens = 0


# ============================================================
# Causal Mask
# ============================================================

def create_causal_mask(seq_len, past_length=0, device=None, dtype=None):
    total_len = seq_len + past_length
    mask = torch.full((seq_len, total_len), float('-inf'), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1 + past_length)
    return mask[None, None, :, :]


# ============================================================
# Output
# ============================================================

@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[DynamicCache] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[tuple] = None
