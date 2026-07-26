from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from torch import nn

from .configuration_qwen3_5 import Qwen3_5TextConfig


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Qwen3_5TextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        rope_params = config.rope_parameters
        self.rope_type = rope_params.get("rope_type", "default")
        base = rope_params.get("rope_theta", 1000000.0)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        head_dim = config.head_dim
        dim = int(head_dim * partial_rotary_factor)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0
        self.mrope_section = rope_params.get("mrope_section", [11, 11, 10])

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


def rotate_half(x: torch.Tensor) -> torch.Tensor:
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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
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
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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

    core_attn_out = torch.zeros(
        batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device
    )
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


class SimpleKVCache:
    """KV cache for full-attention layers. Uses list-based storage (torch.export friendly).

    Mirrors transformers' DynamicCache design: list of layers, each storing
    keys/values as plain tensor attributes. No dict operations.
    """

    def __init__(self):
        self.keys: list[torch.Tensor | None] = []
        self.values: list[torch.Tensor | None] = []

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,
        new_kv_cache_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.keys) <= layer_idx:
            self.keys.append(None)
            self.values.append(None)

        if new_kv_cache_pos is not None and self.keys[layer_idx] is not None:
            # scatter-write: index_copy 到预分配 buffer 的 new_kv_cache_pos 位置 (K=kv_max 固定)
            cache_k = self.keys[layer_idx].index_copy(2, new_kv_cache_pos, key_states)
            cache_v = self.values[layer_idx].index_copy(2, new_kv_cache_pos, value_states)
            self.keys[layer_idx] = cache_k
            self.values[layer_idx] = cache_v
        elif self.keys[layer_idx] is None:
            self.keys[layer_idx] = key_states
            self.values[layer_idx] = value_states
        else:
            # fallback: torch.cat 累积 (K=kv_max+seq_len, 非 scatter, eager generate 用)
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], key_states], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], value_states], dim=2)

        return self.keys[layer_idx], self.values[layer_idx]

    def get_seq_length(self) -> int:
        for i in range(len(self.keys)):
            if self.keys[i] is not None:
                return self.keys[i].shape[2]
        return 0


class SimpleDeltaNetCache:
    def __init__(self, num_layers: int, conv_dim: int, conv_kernel_size: int,
                 k_head_dim: int, v_head_dim: int, num_v_heads: int, batch_size: int = 1,
                 device: torch.device = None, dtype: torch.dtype = torch.float32):
        self.num_layers = num_layers
        self.layers = []
        for _ in range(num_layers):
            self.layers.append({
                "conv_states": torch.zeros(batch_size, conv_dim, conv_kernel_size, device=device, dtype=dtype),
                "recurrent_states": torch.zeros(batch_size, num_v_heads, k_head_dim, v_head_dim, device=device, dtype=dtype),
            })
        self._has_state = False

    def has_previous_state(self, layer_idx: int) -> bool:
        return self._has_state

    def update_conv_state(self, new_state: torch.Tensor, layer_idx: int):
        self.layers[layer_idx]["conv_states"] = new_state
        self._has_state = True

    def update_recurrent_state(self, new_state: torch.Tensor | None, layer_idx: int):
        if new_state is not None:
            self.layers[layer_idx]["recurrent_states"] = new_state


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
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=self.layer_norm_epsilon)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: SimpleDeltaNetCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx]["conv_states"]
            recurrent_state = cache_params.layers[self.layer_idx]["recurrent_states"]

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1:
            mixed_qkv = torch_causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
            )
        else:
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

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

        output = self.out_proj(core_attn_out)
        return output


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

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: SimpleKVCache | None = None,
        new_kv_cache_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, new_kv_cache_pos)

        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)

        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: SimpleKVCache | None = None,
        delta_cache: SimpleDeltaNetCache | None = None,
        new_kv_cache_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=delta_cache,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                new_kv_cache_pos=new_kv_cache_pos,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def _create_causal_mask(
    attention_mask: torch.Tensor | None,
    inputs_embeds: torch.Tensor,
    past_key_values: SimpleKVCache | None = None,
) -> torch.Tensor:
    """Build an additive causal mask, (b, 1, seq_len, total_len).

    `attention_mask=None` produces a pure causal mask (no padding). With a
    2D padding mask the unmasked positions are 0.0 and masked positions are
    `finfo.min`. Decode steps (seq_len==1 with cache) collapse to all-zero
    (single query attends to all cached keys), which is a no-op.

    If `attention_mask` is already 4D, it is returned as-is (pre-constructed).
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask

    batch_size, seq_len = inputs_embeds.shape[:2]
    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    total_len = past_len + seq_len

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    min_val = torch.finfo(dtype).min

    q_pos = torch.arange(seq_len, device=device, dtype=torch.long) + past_len  # (seq_len,)
    k_pos = torch.arange(total_len, device=device, dtype=torch.long)          # (total_len,)
    causal = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)                          # (seq_len, total_len)
    causal_mask = torch.where(causal, torch.tensor(0.0, dtype=dtype, device=device),
                              torch.tensor(min_val, dtype=dtype, device=device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, total_len)

    if attention_mask is not None:
        extended_mask = attention_mask[:, None, None, :].to(dtype=dtype)
        extended_mask = (1.0 - extended_mask) * min_val
        causal_mask = causal_mask + extended_mask

    return causal_mask


class Qwen3_5TextModel(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: SimpleKVCache | None = None,
        delta_cache: SimpleDeltaNetCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool = False,
        new_kv_cache_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SimpleKVCache | None, SimpleDeltaNetCache | None]:
        if (input_ids is None) and (inputs_embeds is None):
            raise ValueError("You must specify either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]

        if use_cache and past_key_values is None:
            past_key_values = SimpleKVCache()
        if use_cache and delta_cache is None:
            # 自动创建 delta_cache (linear attention recurrent state), 对齐 generate 逻辑
            first_linear_layer_idx = next(
                (i for i, lt in enumerate(self.config.layer_types)
                 if lt == "linear_attention"), None)
            if first_linear_layer_idx is not None:
                gdn = self.layers[first_linear_layer_idx].linear_attn
                delta_cache = SimpleDeltaNetCache(
                    num_layers=self.config.num_hidden_layers,
                    conv_dim=gdn.conv_dim,
                    conv_kernel_size=gdn.conv_kernel_size,
                    k_head_dim=gdn.head_k_dim,
                    v_head_dim=gdn.head_v_dim,
                    num_v_heads=gdn.num_v_heads,
                    batch_size=batch_size,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )

        if position_ids is None:
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
            position_ids = position_ids.view(1, 1, -1).expand(4, batch_size, -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        # Causal mask for full-attention layers. Build it for prefill (and the
        # first step); a single-token decode step with an active cache needs no
        # mask (the single query may attend to all cached keys).
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if seq_len == 1 and past_len > 0:
            causal_mask = None
        else:
            causal_mask = _create_causal_mask(attention_mask, inputs_embeds, past_key_values)

        linear_attn_mask = attention_mask
        # decode 步 (有 cache) 不需要 linear mask; prefill 步保留 mask
        # 注意: 原始代码用 torch.all(mask==1) 检查是否全 1 (无 padding),
        # 但 torch.all 是数据依赖的, torch.export 无法追踪, 这里简化为仅检查 cache 状态
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            linear_attn_mask = None

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                delta_cache=delta_cache,
                new_kv_cache_pos=new_kv_cache_pos,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values, delta_cache


class Qwen3_5ForCausalLM(nn.Module):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: SimpleKVCache | None = None,
        delta_cache: SimpleDeltaNetCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        new_kv_cache_pos: torch.Tensor | None = None,
    ) -> dict:
        hidden_states, past_key_values, delta_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            delta_cache=delta_cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            new_kv_cache_pos=new_kv_cache_pos,
        )
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
            "delta_cache": delta_cache,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
        eos_token_id: int | list[int] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = input_ids.device
        past_kv = SimpleKVCache()

        first_linear_layer_idx = None
        for i, lt in enumerate(self.config.layer_types):
            if lt == "linear_attention":
                first_linear_layer_idx = i
                break

        if first_linear_layer_idx is not None:
            gdn = self.model.layers[first_linear_layer_idx].linear_attn
            delta_cache = SimpleDeltaNetCache(
                num_layers=self.config.num_hidden_layers,
                conv_dim=gdn.conv_dim,
                conv_kernel_size=gdn.conv_kernel_size,
                k_head_dim=gdn.head_k_dim,
                v_head_dim=gdn.head_v_dim,
                num_v_heads=gdn.num_v_heads,
                batch_size=input_ids.shape[0],
                device=device,
                dtype=next(self.parameters()).dtype,
            )
        else:
            delta_cache = None

        out = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            delta_cache=delta_cache,
            use_cache=True,
        )
        past_kv = out["past_key_values"]
        delta_cache = out["delta_cache"]

        if eos_token_id is not None and isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for _ in range(max_new_tokens):
            next_logits = out["logits"][:, -1, :]
            if do_sample and temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(attention_mask.shape[0], 1, device=device, dtype=attention_mask.dtype),
                ], dim=-1)

            out = self(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                delta_cache=delta_cache,
                use_cache=True,
            )
            past_kv = out["past_key_values"]
            delta_cache = out["delta_cache"]

            if eos_token_id is not None:
                if any((next_token == eid).all().item() for eid in eos_token_id):
                    break

        return input_ids


def _remap_key(k: str) -> str:
    """Map HuggingFace safetensors key to this implementation's param name.

    VLM checkpoints prefix the text model with `model.language_model.` while
    this standalone text CausalLM uses `model.`. Visual / mtp weights are
    dropped (not present in the text model).
    """
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    return k


def load_qwen3_5_weights(
    model: Qwen3_5ForCausalLM,
    model_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3_5ForCausalLM:
    from safetensors.torch import load_file

    param_dict = dict(model.named_parameters())
    loaded = set()

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif os.path.isfile(single_path):
        shards = ["model.safetensors"]
    else:
        # Sharded single-file naming like `model.safetensors-00001-of-00001.safetensors`
        import glob
        shards = sorted(glob.glob(os.path.join(model_path, "model.safetensors-*.safetensors")))
        if not shards:
            from huggingface_hub import snapshot_download
            cached = snapshot_download(model_path, allow_patterns=["*.safetensors", "*.json"])
            return load_qwen3_5_weights(model, cached, device, dtype)
        shards = [os.path.basename(s) for s in shards]

    for shard in shards:
        shard_path = os.path.join(model_path, shard)
        if not os.path.isfile(shard_path):
            from huggingface_hub import hf_hub_download
            shard_path = hf_hub_download(model_path, shard)
        state = load_file(shard_path, device="cpu")
        remapped = {}
        for k, v in state.items():
            nk = _remap_key(k)
            if nk in param_dict:
                p = param_dict[nk]
                if p.shape != v.shape:
                    print(f"[WARN] shape mismatch {nk}: model {tuple(p.shape)} vs ckpt {tuple(v.shape)}, skip")
                    continue
                remapped[nk] = v.to(device=device, dtype=dtype)
        model.load_state_dict(remapped, strict=False)
        loaded.update(remapped.keys())

    missing = set(param_dict) - loaded
    # lm_head is expected to be missing when tie_word_embeddings (not stored in ckpt)
    if model.config.tie_word_embeddings and "lm_head.weight" in missing:
        missing.discard("lm_head.weight")
    if missing:
        print(f"[WARN] {len(missing)} keys missing (e.g. {list(missing)[:3]})")

    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    return model
