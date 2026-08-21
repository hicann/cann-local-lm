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

from ..fused_ops import ops as _fused_ops  # noqa: F401  (registers cann_lm fused ops)
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


def scatter_kv_to_cache(
    cache: torch.Tensor,
    kv_pos: torch.Tensor,
    kv_new: torch.Tensor,
) -> torch.Tensor:
    """Scatter new KV into BNSD cache via npu_scatter_nd_update (out-of-place, 标准多维坐标).

    cache [b,n,kv_max,d], kv_pos [seq] (dim2 写入位置), kv_new [b,n,seq,d].
    构造 [b*n*seq,3] indices (b,n,kv_pos) + [b*n*seq,d] updates, 命中 ge.ScatterNdUpdate
    标准语义, 等价 index_copy(2, kv_pos, kv_new). CPU (else 分支逐点写) 与 GE
    (ge.ScatterNdUpdate) 语义一致, 不依赖 1D=dim2 私有约定.

    用 out-of-place 版 (无 _): in-place schema 有 alias annotation (Tensor(a!)),
    torch.export functionalization 不支持; out-of-place (-> Tensor out) 无 alias.

    indices 构造避免 Range(arange)+BroadcastTo(expand), 改用 support_ops.md 算子:
      n_grid: Const [n, kv_max] (python list 预计算, n/kv_max 静态 → Const, 无 Range)
      col_n:  n_grid[:, :seq].reshape(-1) (StridedSliceD 动态 slice seq, 无 expand;
              col_n 需 n_idx 每个重复 seq 次, seq 动态, concat 无法替代, 故用 slice)
      col_pos: cat([kv_pos]*n) (ConcatV2D, n 静态 tile, 无 expand)
      col_b:  col_n * 0 (Mul, b=1 全 0; gemma4 batch=1, b>1 需扩展 b 网格)
    """
    b, n, kv_max, d = cache.shape
    seq = kv_pos.shape[0]
    n_grid = torch.tensor(
        [[i] * kv_max for i in range(n)], device=cache.device, dtype=torch.int64
    )
    col_n = n_grid[:, :seq].reshape(-1)
    col_pos = torch.cat([kv_pos] * n, dim=0)
    col_b = col_n * 0
    indices = torch.cat([col_b.view(1, -1), col_n.view(1, -1), col_pos.view(1, -1)], dim=0).transpose(0, 1).reshape(-1, 3)
    updates = kv_new.reshape(-1, d)
    return torch.ops.npu.npu_scatter_nd_update(cache, indices, updates)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q+k pair. 与图灵参考实现一致:

    - sliding (全旋, head_dim == cos.shape[-1]): npu_rotary_mul 单 tensor, q/k 各调一次.
    - full (proportional partial, rotary_dim == cos.shape[-1] < head_dim): cat 出 q_rot/k_rot
      (前 pair_dim + 中 half_dim:half_dim+pair_dim), npu_rotary_mul q/k 各调一次,
      拆装插回原位 (中间和后面保持原值).
    """
    cos = cos.unsqueeze(2)  # [b,s,1,d] -> broadcast to heads (BSND: [b,s,N,d])
    sin = sin.unsqueeze(2)
    if layer_type == "sliding_attention":
        return _rope_sliding(q, k, cos, sin)
    return _rope_full_pair(q, k, cos, sin)


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    layer_type: str,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (for kv_shared path: q 单独, k 从 shared 来已 RoPE).

    sliding: npu_rotary_mul; full: cat/拆装 + npu_rotary_mul (单 tensor 模拟 partial).
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    if layer_type == "sliding_attention":
        return torch.ops.npu.npu_rotary_mul(x, cos, sin, rotary_mode="half")
    return _rope_full_single(x, cos, sin)


def _rope_sliding(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.ops.npu.npu_rotary_mul(q, cos, sin, rotary_mode="half")
    k_out = torch.ops.npu.npu_rotary_mul(k, cos, sin, rotary_mode="half")
    return q_out, k_out


def _rope_full_pair(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_dim = cos.shape[-1] // 2
    half_dim = q.shape[-1] // 2
    q_rot = torch.cat([q[..., :pair_dim], q[..., half_dim:half_dim + pair_dim]], dim=-1)
    k_rot = torch.cat([k[..., :pair_dim], k[..., half_dim:half_dim + pair_dim]], dim=-1)
    q_rot = torch.ops.npu.npu_rotary_mul(q_rot, cos, sin, rotary_mode="half")
    k_rot = torch.ops.npu.npu_rotary_mul(k_rot, cos, sin, rotary_mode="half")
    q_out = torch.cat(
        [q_rot[..., :pair_dim], q[..., pair_dim:half_dim],
         q_rot[..., pair_dim:], q[..., half_dim + pair_dim:]], dim=-1,
    )
    k_out = torch.cat(
        [k_rot[..., :pair_dim], k[..., pair_dim:half_dim],
         k_rot[..., pair_dim:], k[..., half_dim + pair_dim:]], dim=-1,
    )
    return q_out, k_out


def _rope_full_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    pair_dim = cos.shape[-1] // 2
    half_dim = x.shape[-1] // 2
    x_rot = torch.cat([x[..., :pair_dim], x[..., half_dim:half_dim + pair_dim]], dim=-1)
    x_rot = torch.ops.npu.npu_rotary_mul(x_rot, cos, sin, rotary_mode="half")
    return torch.cat(
        [x_rot[..., :pair_dim], x[..., pair_dim:half_dim],
         x_rot[..., pair_dim:], x[..., half_dim + pair_dim:]], dim=-1,
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: (b, kv_heads, s, d) -> (b, heads, s, d).

    用 reshape+cat+view (npu_tuned 方案) 替代 expand, 使导出图里是 ConcatV2 而非 BroadcastTo。
    先 reshape 到 [b*kv, 1, s, d] (kv 并入 dim0, dim1 留 size-1 给 cat), 再 cat([x]*n_rep, dim=1)
    物化 n_rep 份, 最后 view 回 [b, kv*n_rep, s, d]。头序 kv_head-major (kv 外层, n_rep 内层),
    与 expand+reshape 逐元素一致; naive cat([x]*n_rep, dim=1) 是 rep-major 会与 q 错位, 不可用。
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states.reshape(batch * num_kv_heads, 1, slen, head_dim)
    hidden_states = torch.cat([hidden_states] * n_rep, dim=1)
    return hidden_states.view(batch, num_kv_heads * n_rep, slen, head_dim)


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
    """Eager (non-flash) scaled dot-product attention with optional soft-capping.

    精度策略 (已验证, 不要改):
    - Q@K matmul: 保持 bfloat16 (和 HF 一致, BLAS 舍入最接近)
      float32 Q@K 反而更差: 和 HF 的 bfloat16 差异更大, 第一个 token 匹配率从 78% 降到 33%
    - softmax: 用 float32 (npu_tuned_export 方案, 已有)
      bfloat16 softmax 在 [6, 2048] (scatter-write buffer) 下有精度损失
    - @v matmul: 保持 bfloat16 (已验证无差异)
    scatter-write 下 head_dim=512 的 full_attention 层有 0.25 BLAS 舍入差异, 但无法消除
    (float32 会引入更大差异), 7/9 第一个 token 与 HF 一致, 可接受
    """
    if scaling is None:
        scaling = module.head_dim ** -0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        inv_softcap = torch.tensor(1.0 / softcap, dtype=attn_weights.dtype, device=attn_weights.device)
        softcap_t = torch.tensor(softcap, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = attn_weights * inv_softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap_t

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def _pad_mask_to_kv_square(mask: torch.Tensor, kv_len: int) -> torch.Tensor:
    """pad atten_mask 的 q 维到 kv_len (融合算子要求 [*, kv_len, kv_len] 方阵).

    cann_lm 的 mask 是 [*, q_len, kv_len] (q-kv mask, q 维=q_len). 融合算子要求
    [*, kv_len, kv_len] (全 KV buffer 因果模板). pad q 维到 kv_len: 前 q_len 行原 mask
    (bool: False=attend, True=mask out), 后 kv_len-q_len 行 pad True (mask out, 不影响,
    融合算子用 actual_seq_qlen 只取前 q_len 行).
    """
    q_len = mask.shape[-2]
    if q_len >= kv_len:
        return mask.bool()
    pad_shape = (*mask.shape[:-2], kv_len - q_len, kv_len)
    pad = torch.full(pad_shape, True, dtype=torch.bool, device=mask.device)
    return torch.cat([mask.bool(), pad], dim=-2)


def fused_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    new_kv_cache_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """融合 attention via npu_fused_infer_attention_score_v2 (替代 eager_attention_forward).

    与图灵参考实现一致: 融合 QK^T+softmax+AV+GQA, 消除 repeat_kv; sparse_mode+pre_tokens
    表达 causal/sliding, 消除 _prepare_causal_mask 的 4D mask 构造. q/k/v 为 BNSD [b,n,s,d]
    (RoPE 后已 transpose). GQA 由 num_query_heads vs num_key_value_heads 表达 (融合算子原生).

    actual_seq_kvlen 用 new_kv_cache_pos.max()+1 算实际有效 kv (past+new), 不能用 key.shape[2]
    (scatter-write 后的全 buffer, 含未初始化部分, 会让融合算子 attend 到垃圾).
    """
    q_len = query.shape[2]
    # actual_kv_len: 实际有效 kv 长度 (past + new).
    # - scatter-write 模式 (OM export): new_kv_cache_pos 不为 None, 用 max+1 (写入位置+1, 不含 buffer 未用部分)
    # - torch.cat 模式 (model.generate eager): new_kv_cache_pos=None, key.shape[2]=past+q_len (实际有效)
    if new_kv_cache_pos is not None:
        actual_kv_len = torch.tensor(q_len, dtype=torch.int64, device=query.device)
    else:
        actual_kv_len = torch.tensor(key.shape[2], dtype=torch.int64, device=query.device)
    actual_seq_qlen = torch.tensor([q_len], dtype=torch.int64, device=query.device)
    actual_seq_kvlen = actual_kv_len.reshape(1).to(torch.int64)
    # atten_mask: sparse_mode=3/4 要求 atten_mask 非空 (bool [*, kv_len, kv_len] 方阵).
    # cann_lm float mask [*, q_len, kv_len] (0.0=attend k<=q, min=mask out k>q) -> bool 转换 +
    # pad q 维到 kv_len (融合算子用 actual_seq_qlen 切片前 q_len 行). 试不取反 (False=attend).
    atten_mask_bool = _pad_mask_to_kv_square(attention_mask, key.shape[2]) if attention_mask is not None else None
    if module.is_sliding:
        sparse_mode, pre_tokens = 4, module.sliding_window
    else:
        sparse_mode = 3
        pre_tokens = torch.iinfo(torch.int32).max
    attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
        query, key, value,
        num_query_heads=module.config.num_attention_heads,
        num_key_value_heads=module.config.num_key_value_heads,
        softmax_scale=module.scaling,
        input_layout="BNSD",
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=0,
        atten_mask=atten_mask_bool,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    # 融合算子输出 BNSD [b,n,s,d], Attention.forward 期望 BSND [b,s,n,d] (reshape [b,s,-1] 假设 BSND).
    # transpose(1,2) BNSD->BSND, 跟 eager_attention_forward 输出一致.
    return attn_output.transpose(1, 2).contiguous()


# ════════════════════════════════════════════════════════════════════
# RMSNorm
# ════════════════════════════════════════════════════════════════════

class Gemma4RMSNorm(nn.Module):
    """RMSNorm matching the Gemma-4 convention, via cann_lm::rms_norm (fused)."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            # gamma is identically 1; pre-allocate a ones buffer so forward can pass
            # a weight tensor to the fused op without allocating each call.
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = torch.ops.npu.npu_rms_norm(x, self.weight, epsilon=self.eps)
        return y

    def forward_add(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """add + rms_norm (拆开): returns (rms_norm(x1 + x2), x1 + x2).

        原用 fused npu_add_rms_norm, 现拆为 add + npu_rms_norm (用户要求).
        """
        residual = x1 + x2
        y, _ = torch.ops.npu.npu_rms_norm(residual, self.weight, epsilon=self.eps)
        return y, residual

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

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

        dev = device if device is not None else torch.device("cpu")
        max_seq = config.max_position_embeddings

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

            # 预缓存 cos/sin 表 (init 算: inv_freq + outer + cat + cos/sin, 全在 init, 不进图)。
            # forward 只 Gather (cos_cached[position_ids]), 消除 Pow/Neg/Cos/Sin 动态算子
            # (npu_tuned 2.3: 预缓存表 + Gather over 常量表)。
            inv_freq = self._compute_inv_freq(layer_type, dev, torch.float32)
            t = torch.arange(max_seq, device=dev, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer(f"cos_cached_{layer_type}", emb.cos(), persistent=False)
            self.register_buffer(f"sin_cached_{layer_type}", emb.sin(), persistent=False)

    def _compute_inv_freq(self, layer_type: str, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        rope_config = self._rope_configs[layer_type]
        rope_type = rope_config["rope_type"]
        base = rope_config["base"]
        partial_rotary_factor = rope_config["partial_rotary_factor"]
        head_dim = rope_config["head_dim"]

        if rope_type == "proportional":
            # 与图灵参考实现一致: inv_freq 长度 = rope_angles = rotary_dim//2 (不补 0).
            # cos/sin 长度 = 2*rope_angles = rotary_dim, 由 apply_rotary_pos_emb 的
            # cat/拆装路径 (full 层) 消费, 不再用 inv_freq 后半补 0 等效 partial rotary.
            rope_angles = int(partial_rotary_factor * head_dim // 2)
            inv_head_dim = torch.tensor(1.0 / head_dim, dtype=dtype, device=device)
            exponents = torch.arange(0, 2 * rope_angles, 2, dtype=dtype, device=device) * inv_head_dim
            return torch.pow(base, -exponents)
        else:
            dim = int(head_dim * partial_rotary_factor)
            inv_dim = torch.tensor(1.0 / dim, dtype=dtype, device=device)
            exponents = torch.arange(0, dim, 2, dtype=dtype, device=device) * inv_dim
            return torch.pow(base, -exponents)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str = "full_attention"):
        # Gather over 预缓存常量表 (cos_cached[position_ids])。无 inv_freq/cos/sin/pow/neg 动态算子。
        cos_cached = getattr(self, f"cos_cached_{layer_type}")  # [max_seq, dim] 常量
        sin_cached = getattr(self, f"sin_cached_{layer_type}")
        # index_select 替代 advanced indexing (cos_cached[position_ids] -> ge.Index, 未在
        # support_ops.md 列出); index_select(0, idx) -> ge.GatherV2 (support_ops.md GatherV2).
        pos_flat = position_ids.reshape(-1)
        cos = cos_cached.index_select(0, pos_flat).reshape(*position_ids.shape, -1)
        sin = sin_cached.index_select(0, pos_flat).reshape(*position_ids.shape, -1)
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
        new_kv_cache_pos: torch.Tensor | None = None,
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

        # --- KV projection (or reuse shared) + RoPE ---
        # 与图灵参考实现一致: 非 kv_shared 用配对 RoPE (sliding/full: npu_rotary_mul q/k 各调
        # + cat/拆装). kv_shared 路径 q 单独 RoPE
        # (k 从 shared 来已 RoPE), 用 single 版 (npu_rotary_mul + cat/拆装 模拟 partial).
        if self.is_kv_shared_layer and shared_kv_states is not None:
            kv_key = self.layer_type
            if kv_key not in shared_kv_states:
                kv_key = next(iter(shared_kv_states), None)
            if kv_key is not None:
                key_states, value_states = shared_kv_states[kv_key]
                key_states = key_states.to(query_states.device)
                value_states = value_states.to(query_states.device)
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, self.layer_type)
            query_states = query_states.transpose(1, 2)
        else:
            key_states = self.k_proj(hidden_states).view(kv_shape)
            key_states = self.k_norm(key_states)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, self.layer_type)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

            if self.v_proj is not None:
                value_states = self.v_proj(hidden_states).view(kv_shape)
            else:
                value_states = key_states
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        # --- KV cache update (scatter-write, fixed shape) ---
        if past_key_values is not None and not self.is_kv_shared_layer:
            cache_entry = past_key_values[self.layer_idx]
            if cache_entry is not None and new_kv_cache_pos is not None:
                cache_k, cache_v = cache_entry
                # Scatter-write new KV into BNSD cache via npu_scatter_nd_update (标准多维坐标).
                # ge.ScatterNdUpdate 按多维坐标 scatter: scatter_kv_to_cache 构造 [b*n*seq,3]
                # indices (b,n,kv_pos) + [b*n*seq,d] updates, 等价 index_copy(2,pos,kv_new).
                # CPU (else 分支) 与 GE (ge.ScatterNdUpdate) 语义一致, 不依赖 1D=dim2 约定.
                # cann_lm 自带 converter (converter/export/_scatter_nd_update_converter.py)
                # 覆盖 torchair, 直接产出 ge.ScatterNdUpdate (不绕 InplaceUpdate+Transpose).
                cache_k = scatter_kv_to_cache(cache_k, new_kv_cache_pos, key_states)
                cache_v = scatter_kv_to_cache(cache_v, new_kv_cache_pos, value_states)
                past_key_values[self.layer_idx] = (cache_k, cache_v)
                # Use entire buffer for attention
                key_states = cache_k
                value_states = cache_v
            elif cache_entry is not None:
                # Fallback: torch.cat (backward compat, cache shape grows)
                key_states = torch.cat([cache_entry[0], key_states], dim=2)
                value_states = torch.cat([cache_entry[1], value_states], dim=2)
                past_key_values[self.layer_idx] = (key_states, value_states)
            else:
                # No cache entry yet, store new KV
                past_key_values[self.layer_idx] = (key_states, value_states)

        if self.store_full_length_kv and shared_kv_states is not None:
            shared_kv_states[self.layer_type] = (key_states, value_states)

        # --- Attention (fused: npu_fused_infer_attention_score_v2) ---
        attn_output = fused_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            new_kv_cache_pos,
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
            shared_kv_states=shared_kv_states,
            new_kv_cache_pos=new_kv_cache_pos,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = self.pre_feedforward_layernorm.forward_add(residual, hidden_states)

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
        new_kv_cache_pos: torch.Tensor | None = None,
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
            # 2D mask: [batch, seq, seq] → [batch, 1, seq, seq] (heads 维 broadcast)
            return torch.where(mask, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(min_dtype, device=device, dtype=dtype)).unsqueeze(1)

        # 4D mask: apply sliding window restriction for sliding_attention layers
        if attention_mask.dim() == 4:
            q_len = attention_mask.shape[-2]
            kv_len = attention_mask.shape[-1]
            if layer_type == "sliding_attention":
                sliding_window = getattr(self.config, "sliding_window", None)
                # q 的 buffer 写入位置: 用 new_kv_cache_pos (与 k_pos 同基准), 而非 arange(q_len).
                # decode 时 arange(1)=[0] != 真实写入位置 (如 8), 会让 window 错误屏蔽所有 k_pos!=0.
                # new_kv_cache_pos 是 q 写进 buffer 的位置, 直接对应 buffer 维度.
                if sliding_window is not None and sliding_window > 0 and kv_len > sliding_window:
                    min_dtype_val = torch.finfo(attention_mask.dtype).min
                    if new_kv_cache_pos is not None:
                        q_pos_buf = new_kv_cache_pos.to(device=attention_mask.device, dtype=attention_mask.dtype)
                    else:
                        q_pos_buf = torch.arange(q_len, device=attention_mask.device, dtype=attention_mask.dtype)
                    k_pos = torch.arange(kv_len, device=attention_mask.device, dtype=attention_mask.dtype)
                    diff = q_pos_buf.unsqueeze(1) - k_pos.unsqueeze(0)
                    window_mask = ~((diff >= 0) & (diff <= sliding_window - 1))
                    attention_mask = attention_mask.masked_fill(
                        window_mask.unsqueeze(0).unsqueeze(0), min_dtype_val
                    )
            return attention_mask

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
        new_kv_cache_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list, dict | None]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        if position_ids is None:
            position_ids = self._prepare_position_ids(inputs_embeds, past_key_values)

        causal_mask = self._prepare_causal_mask(inputs_embeds, attention_mask, new_kv_cache_pos=new_kv_cache_pos)
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

            mask = self._prepare_causal_mask(inputs_embeds, attention_mask, layer_type, new_kv_cache_pos=new_kv_cache_pos)

            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            hidden_states = layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=pos_emb,
                attention_mask=mask,
                past_key_values=past_key_values,
                new_kv_cache_pos=new_kv_cache_pos,
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
        new_kv_cache_pos: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> dict:
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            shared_kv_states=shared_kv_states,
            output_hidden_states=output_hidden_states,
            new_kv_cache_pos=new_kv_cache_pos,
            position_ids=position_ids,
        )
        if output_hidden_states:
            hidden_states, past_key_values, shared_kv_states, all_hidden_states = model_output
        else:
            hidden_states, past_key_values, shared_kv_states = model_output
            all_hidden_states = None
        logits = self.lm_head(hidden_states)

        if self.final_logit_softcapping is not None:
            inv_softcap = torch.tensor(1.0 / self.final_logit_softcapping, dtype=logits.dtype, device=logits.device)
            softcap = torch.tensor(self.final_logit_softcapping, dtype=logits.dtype, device=logits.device)
            logits = logits * inv_softcap
            logits = torch.tanh(logits)
            logits = logits * softcap

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
    "apply_rotary_pos_emb_single",
    "repeat_kv",
    "eager_attention_forward",
    "fused_attention_forward",
    "load_gemma4_weights",
]
