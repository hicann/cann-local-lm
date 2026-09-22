# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""cann_lm::fused_infer_attention_score_v2 - fused attention (schema + meta + CPU eager kernel).

注册到 npu 命名空间 (torch.ops.cann_lm.fused_infer_attention_score_v2), torchair custom/
的 fused_infer_attention_score_v2 converter 在 air 命名空间注册 (不适用于 npu 命名空间),
故 cann_lm 在 ge_converter.py 自己注册 npu 命名空间 converter (调 ge.FusedInferAttentionScore).

参考实现 (cann-recipes-infer): sliding 层 TND + sparse_mode=4 (sliding_window),
full 层 BNSD(prefill)/BSH(decode) + sparse_mode=3(causal)/0(decode). 融合算子原生支持
GQA (num_query_heads vs num_key_value_heads), 消除 repeat_kv; sparse_mode + pre_tokens
表达 causal/sliding, 消除 _prepare_causal_mask 的 4D mask 构造.

schema 简化: 只保留 gemma4 用的参数 (去掉 quant/antiquant/rope/shared_prefix/pse_shift 等).
CPU eager: 简化版 (BSND/BNSD + sparse_mode 0/3/4 + 无 quant/paged), 用于数值验证.
"""

import torch

from ._library import NPU_LIB, safe_define, _HAS_TORCH_NPU, register_cpu_impl

_FA_SCHEMA = (
    "fused_infer_attention_score_v2(Tensor query, Tensor key, Tensor value, *, "
    "Tensor? atten_mask=None, Tensor? actual_seq_qlen=None, Tensor? actual_seq_kvlen=None, "
    "Tensor? block_table=None, "
    "int num_query_heads=1, int num_key_value_heads=0, float softmax_scale=1.0, "
    "int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout=\"BSH\", "
    "int sparse_mode=0, int block_size=0, int inner_precise=0, bool return_softmax_lse=False"
    ") -> (Tensor attention_out, Tensor softmax_lse)"
)

if NPU_LIB is not None:
    safe_define(_FA_SCHEMA)

if not _HAS_TORCH_NPU:

    @torch.library.register_fake("cann_lm::fused_infer_attention_score_v2")
    def _fa_meta(
        query, key, value, *,
        atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None, block_table=None,
        num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0,
        pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH",
        sparse_mode=0, block_size=0, inner_precise=0, return_softmax_lse=False,
    ):
        # output shape 跟 query 同 (除 layout 转换外, 融合算子保持 q 的 batch/seq/heads/dim)
        out = torch.empty_like(query)
        lse = torch.empty([0], dtype=torch.float32, device="meta")
        return out, lse

    @register_cpu_impl("cann_lm::fused_infer_attention_score_v2")
    def _fa_cpu(
        query, key, value, *,
        atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None, block_table=None,
        num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0,
        pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH",
        sparse_mode=0, block_size=0, inner_precise=0, return_softmax_lse=False,
    ):
        """CPU eager: 简化版 attention (BSND/BNSD/BSH/TND + sparse_mode 0/3/4 + 无 quant/paged).

        数值与 eager_attention_forward 等价 (matmul + softmax + matmul + GQA + mask).
        复杂场景 (paged/quant/antiquant) raise NotImplementedError.
        """
        if block_table is not None:
            raise NotImplementedError("PagedAttention (block_table) not supported on CPU")
        if num_key_value_heads == 0:
            num_key_value_heads = num_query_heads

        q, k, v = _to_bnsd(query, key, value, input_layout, num_query_heads, num_key_value_heads)
        # GQA: repeat_kv
        n_rep = num_query_heads // num_key_value_heads
        if n_rep > 1:
            k = _repeat_kv(k, n_rep)
            v = _repeat_kv(v, n_rep)
        # Q@K^T * scale
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * softmax_scale
        # mask: sparse_mode (0=none 3=causal 4=sliding_window) + pre/next_tokens
        causal_mask = _build_sparse_mask(q, sparse_mode, pre_tokens, next_tokens)
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
        if atten_mask is not None:
            # gemma4 mask 方阵 [b,1,kv_max,kv_max] bool (False=attend, True=mask out):
            # 截前 seq_len 行匹配 attn_weights [b,heads,seq_len,kv_max] + bool→float (True→-inf)。
            # 非方阵 [b,1,seq_len,kv_max] float mask (其他模型) 直用, 不受影响。
            if atten_mask.shape[2] != attn_weights.shape[2]:
                atten_mask = atten_mask[:, :, :attn_weights.shape[2], :]
            if atten_mask.dtype == torch.bool:
                min_val = torch.finfo(attn_weights.dtype).min
                atten_mask = torch.where(atten_mask, min_val, 0.0).to(attn_weights.dtype)
            attn_weights = attn_weights + atten_mask
        # softmax (fp32) -> @V
        attn_weights = torch.softmax(attn_weights.float(), dim=-1).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        # 转回 input_layout
        attn_output = _from_bnsd(attn_output, input_layout, query)
        lse = torch.empty([0], dtype=torch.float32, device=query.device)
        return attn_output, lse


def _to_bnsd(q, k, v, layout, num_q_heads, num_kv_heads):
    """任意 layout -> BNSD [b, n, s, d]. 支持 BSND/BNSD/BSH/TND."""
    if layout == "BNSD":
        return q, k, v
    if layout == "BSND":
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if layout == "BSH":  # [b, s, n*d] -> [b, n, s, d]
        b, s, _ = q.shape
        d = q.shape[-1] // num_q_heads
        q_bnsd = q.view(b, s, num_q_heads, d).transpose(1, 2)
        b, s, _ = k.shape
        d = k.shape[-1] // num_kv_heads
        k_bnsd = k.view(b, s, num_kv_heads, d).transpose(1, 2)
        v_bnsd = v.view(b, s, num_kv_heads, d).transpose(1, 2)
        return q_bnsd, k_bnsd, v_bnsd
    if layout == "TND":  # [t, n, d] -> [1, n, t, d]
        t, _, d = q.shape
        q_bnsd = q.unsqueeze(0)
        k_bnsd = k.unsqueeze(0)
        v_bnsd = v.unsqueeze(0)
        return q_bnsd, k_bnsd, v_bnsd
    raise NotImplementedError(f"layout {layout} not supported on CPU")


def _from_bnsd(x_bnsd, layout, query):
    """BNSD -> 原 layout."""
    if layout == "BNSD":
        return x_bnsd
    if layout == "BSND":
        return x_bnsd.transpose(1, 2)
    if layout == "BSH":
        b, n, s, d = x_bnsd.shape
        return x_bnsd.transpose(1, 2).reshape(b, s, n * d)
    if layout == "TND":
        return x_bnsd.squeeze(0)
    raise NotImplementedError(f"layout {layout} not supported on CPU")


def _repeat_kv(x, n_rep):
    """GQA: [b, kv, s, d] -> [b, kv*n_rep, s, d] (kv_head-major)."""
    b, kv, s, d = x.shape
    x = x.reshape(b * kv, 1, s, d)
    x = torch.cat([x] * n_rep, dim=1)
    return x.view(b, kv * n_rep, s, d)


def _build_sparse_mask(q, sparse_mode, pre_tokens, next_tokens):
    """sparse_mode: 0=none, 3=causal, 4=sliding_window. 返回 [1,1,s,s] mask 或 None."""
    if sparse_mode == 0:
        return None
    b, n, s_q, s_k = q.shape
    if s_q != s_k:
        return None  # decode (q_len != kv_len) 场景由 atten_mask 处理
    q_pos = torch.arange(s_q, device=q.device)
    k_pos = torch.arange(s_k, device=q.device)
    if sparse_mode == 3:  # causal
        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    elif sparse_mode == 4:  # sliding_window
        diff = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)
        mask = (diff >= 0) & (diff <= pre_tokens - 1)
    else:
        return None
    min_val = torch.finfo(q.dtype).min
    return torch.where(mask, torch.tensor(0.0, device=q.device, dtype=q.dtype),
                       torch.tensor(min_val, device=q.device, dtype=q.dtype)).unsqueeze(0).unsqueeze(0)
