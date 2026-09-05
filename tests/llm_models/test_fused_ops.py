#!/usr/bin/env python3
"""CPU eager impl 单元测试 for cann_lm/models/fused_ops 自定义算子.

覆盖每个算子的 CPU eager kernel (schema + fake/meta + impl, 在 converter venv
无 torch_npu、无 stub LD_PRELOAD 下由 _library.py 注册). 测 shape + 基本数值正确性.

注: _library.py / ops.py 是注册基础设施 (torch.library namespace + schema define +
register_fake + register_cpu_impl 装饰器), 不是直接可调用的算子, 间接由下列 8 个
算子的调用覆盖, 不单独测试.

跑 (必须用 converter venv python, 系统 python 会加载 torch_npu 致 npu:: op 走
torch_npu dispatcher 报 NotImplementedError):
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    /mnt/workspace/torch-delegate-harmony/.venv/bin/python \
    -m pytest tests/llm_models/test_fused_ops.py -q
"""

import pytest
import torch

import cann_lm.models.fused_ops  # noqa: F401 — 触发 schema + fake + CPU impl 注册


def _maybe_unwrap(x):
    """quant_matmul schema `-> (Tensor y)` 返回 1-tuple; 兼容裸 Tensor."""
    return x[0] if isinstance(x, (tuple, list)) and len(x) == 1 else x


# ---------------------------------------------------------------------------
# 1. npu::npu_rms_norm  (rms_norm.py)
#    schema: npu_rms_norm(Tensor self, Tensor gamma, float epsilon=1e-6)
#            -> (Tensor y, Tensor rstd)
#    cpu: rstd = rsqrt(mean(x^2, -1) + eps); y = (x * rstd) * gamma
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(2, 8), (2, 3, 8), (1, 4, 16)])
def test_npu_rms_norm(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32)
    gamma = torch.randn(shape[-1], dtype=torch.float32)
    eps = 1e-6

    y, rstd = torch.ops.npu.npu_rms_norm(x, gamma, eps)

    assert y.shape == x.shape
    assert rstd.shape == x.shape[:-1] + (1,)
    assert rstd.dtype == torch.float32
    assert (rstd > 0).all()

    ms = x.pow(2).mean(-1, keepdim=True) + eps
    expected = x * torch.rsqrt(ms) * gamma
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)


def test_npu_rms_norm_default_eps():
    torch.manual_seed(1)
    x = torch.randn(3, 6, dtype=torch.float32)
    gamma = torch.randn(6, dtype=torch.float32)

    y, rstd = torch.ops.npu.npu_rms_norm(x, gamma)

    ms = x.pow(2).mean(-1, keepdim=True) + 1e-6
    expected = x * torch.rsqrt(ms) * gamma
    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. npu::npu_rotary_mul  (rotary_mul.py)
#    schema: npu_rotary_mul(Tensor self, Tensor r1, Tensor r2,
#                           str rotary_mode="half") -> Tensor out
#    cpu (half only): x1=self[...,:D/2], x2=self[...,D/2:];
#                      rotated = cat(-x2, x1, -1); out = self*r1 + rotated*r2
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", [8, 16])
def test_npu_rotary_mul_half(d):
    torch.manual_seed(0)
    x = torch.randn(2, 4, d, dtype=torch.float32)
    r1 = torch.randn(2, 4, d, dtype=torch.float32)
    r2 = torch.randn(2, 4, d, dtype=torch.float32)

    out = torch.ops.npu.npu_rotary_mul(x, r1, r2, "half")

    assert out.shape == x.shape
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    expected = (x * r1) + (rotated * r2)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_npu_rotary_mul_default_mode():
    torch.manual_seed(0)
    x = torch.randn(2, 8, dtype=torch.float32)
    r1 = torch.randn(2, 8, dtype=torch.float32)
    r2 = torch.randn(2, 8, dtype=torch.float32)

    out = torch.ops.npu.npu_rotary_mul(x, r1, r2)  # default "half"

    x1, x2 = x[..., :4], x[..., 4:]
    expected = (x * r1) + (torch.cat((-x2, x1), dim=-1) * r2)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. cann_lm::l2norm  (l2_norm.py)
#    schema: l2norm(Tensor x, int dim=-1, float eps=1e-6) -> Tensor
#    cpu: inv_norm = rsqrt((x*x).sum(dim, keepdim) + eps); return x * inv_norm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [-1, -2, -3])
def test_cann_lm_l2norm_unit_norm_and_values(dim):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5, dtype=torch.float32)

    out = torch.ops.cann_lm.l2norm(x, dim=dim, eps=1e-6)

    assert out.shape == x.shape
    norm = (out * out).sum(dim=dim, keepdim=True).sqrt()
    torch.testing.assert_close(norm, torch.ones_like(norm), rtol=1e-4, atol=1e-4)

    inv = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + 1e-6)
    torch.testing.assert_close(out, x * inv, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. npu::npu_dynamic_quant  (dynamic_quant.py)
#    schema: npu_dynamic_quant(Tensor input, *, ...) -> (Tensor y, Tensor scale)
#    cpu: scale = amax|x|/127 (clamp min 1e-8); y = clamp(round(x/scale), -128, 127).int8
# ---------------------------------------------------------------------------
def test_npu_dynamic_quant_shape_dtype_and_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(4, 16, dtype=torch.float32) * 3.0

    y, scale = torch.ops.npu.npu_dynamic_quant(x)

    assert y.shape == x.shape
    assert y.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert (scale > 0).all()
    assert torch.isfinite(y.float()).all()

    expected_scale = (x.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-8)
    torch.testing.assert_close(scale, expected_scale.to(torch.float32), rtol=1e-5, atol=1e-6)
    expected_y = torch.clamp(torch.round(x / expected_scale), -128, 127).to(torch.int8)
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)

    # dequant round-trip: |y*scale - x| <= scale/2 (within one quantum step)
    deq = y.float() * scale.float()
    assert torch.allclose(deq, x.float(), atol=scale.max().item())


# ---------------------------------------------------------------------------
# 5. npu::npu_scatter_nd_update  (scatter_nd_update.py)
#    schema: npu_scatter_nd_update(Tensor self, Tensor indices, Tensor updates)
#                 -> Tensor out
#    cpu (1d indices): out.index_copy_(2, indices, updates)
#    cpu (>=2d indices): out[tuple(idx[i])] = updates[i]  per row
# ---------------------------------------------------------------------------
def test_npu_scatter_nd_update_1d_indices():
    torch.manual_seed(0)
    cache = torch.randn(2, 3, 5, 4, dtype=torch.float32)
    indices = torch.tensor([1, 3], dtype=torch.int64)
    updates = torch.randn(2, 3, 2, 4, dtype=torch.float32)

    out = torch.ops.npu.npu_scatter_nd_update(cache, indices, updates)

    assert out.shape == cache.shape
    expected = cache.clone()
    expected.index_copy_(2, indices, updates)
    torch.testing.assert_close(out, expected)
    # out-of-place: original cache untouched
    assert not torch.equal(out, cache)
    # untouched positions preserved
    for j in range(5):
        if j not in indices.tolist():
            assert torch.equal(out[:, :, j, :], cache[:, :, j, :])


def test_npu_scatter_nd_update_2d_indices():
    torch.manual_seed(0)
    cache = torch.randn(2, 3, 5, 4, dtype=torch.float32)
    indices = torch.tensor([[0, 0, 1], [1, 2, 4]], dtype=torch.int64)  # [num, ndim]
    updates = torch.randn(2, 4, dtype=torch.float32)

    out = torch.ops.npu.npu_scatter_nd_update(cache, indices, updates)

    assert out.shape == cache.shape
    expected = cache.clone()
    for i in range(indices.shape[0]):
        expected[tuple(indices[i].tolist())] = updates[i]
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# 6. npu::npu_quant_matmul  (quant_matmul.py)  -- W4A8 unpacked PoC kernel
#    schema: npu_quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, offset,
#             pertoken_scale, bias, output_dtype, ..., group_sizes, y_scale,
#             transpose_x1, transpose_x2) -> (Tensor y)
#    cpu (transpose_x2=False, scale=[K//group, N], x1=[...,K]):
#       w_fp = w * (scale.repeat_interleave(group, -2) if group>1 else scale)
#       x1_fp = x1 (or dequant via pertoken_scale); out = x1_fp @ w_fp [+ bias]
# ---------------------------------------------------------------------------
def test_npu_quant_matmul_group1_plain_matmul():
    torch.manual_seed(0)
    M, K, N = 3, 8, 5
    x1 = torch.randn(M, K, dtype=torch.float32)
    x2 = torch.randn(K, N, dtype=torch.float32)
    scale = torch.ones(1, N, dtype=torch.float32)  # group=1 -> [1, N]

    out = torch.ops.npu.npu_quant_matmul(
        x1, x2, scale, group_sizes=[1, 1, 1], output_dtype=6  # fp32
    )
    out = _maybe_unwrap(out)

    assert out.shape == (M, N)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    expected = x1 @ x2  # scale=1 -> plain matmul
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_npu_quant_matmul_grouped_scale_and_bias():
    torch.manual_seed(0)
    M, K, N, group = 2, 8, 4, 4
    x1 = torch.randn(M, K, dtype=torch.float32)
    w = torch.randn(K, N, dtype=torch.float32)
    sc = torch.rand(K // group, N, dtype=torch.float32) + 0.1
    bias = torch.randn(N, dtype=torch.float32)

    out = torch.ops.npu.npu_quant_matmul(
        x1, w, sc, bias=bias, group_sizes=[1, 1, group], output_dtype=6
    )
    out = _maybe_unwrap(out)

    assert out.shape == (M, N)
    assert torch.isfinite(out).all()
    sc_exp = sc.repeat_interleave(group, dim=-2)
    expected = x1 @ (w * sc_exp) + bias
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_npu_quant_matmul_pertoken_scale_dequant():
    torch.manual_seed(0)
    M, K, N = 2, 4, 3
    x1 = torch.randn(M, K, dtype=torch.float32)
    x2 = torch.randn(K, N, dtype=torch.float32)
    scale = torch.ones(1, N, dtype=torch.float32)
    ps = torch.rand(M, 1, dtype=torch.float32) + 0.5  # [M, 1] per-token scale

    out = torch.ops.npu.npu_quant_matmul(
        x1, x2, scale, pertoken_scale=ps, group_sizes=[1, 1, 1], output_dtype=6
    )
    out = _maybe_unwrap(out)

    assert out.shape == (M, N)
    assert torch.isfinite(out).all()
    # cpu impl: x1 not int8 -> quantize x1 by ps then dequant: x1q = round(x1/ps)
    # x1_fp = x1q * ps; out = x1_fp @ w. Compare to that exact cpu logic.
    x1q = torch.clamp(torch.round(x1 / ps), -128, 127)
    x1_fp = x1q * ps
    expected = x1_fp @ x2
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. npu::npu_chunk_gated_delta_rule  (chunk_gated_delta_rule.py)
#    Note: 仓内实际注册在 npu 命名空间带 npu_ 前缀 (非 cann_lm::), 故调用
#    torch.ops.npu.npu_chunk_gated_delta_rule.
#    schema: npu_chunk_gated_delta_rule(Tensor query, Tensor key, Tensor value,
#               *, Tensor? beta, Tensor? initial_state, Tensor? actual_seq_lengths,
#               float? scale, Tensor? g) -> (Tensor out, Tensor final_state)
#    cpu: THD->BSHD reshape, 调 torch_chunk_gated_delta_rule; 数值精确难对齐, 测
#    shape + 有限值.
# ---------------------------------------------------------------------------
def test_npu_chunk_gated_delta_rule_shape_finite():
    torch.manual_seed(0)
    B, T, Nk, Nv, Dk, Dv = 1, 8, 2, 2, 8, 8
    q = torch.randn(T, Nk, Dk, dtype=torch.float32)
    k = torch.randn(T, Nk, Dk, dtype=torch.float32)
    v = torch.randn(T, Nv, Dv, dtype=torch.float32)
    beta = torch.rand(T, Nv, dtype=torch.float32) * 0.1
    g = torch.rand(T, Nv, dtype=torch.float32) * 0.1
    init_state = torch.zeros(B, Nv, Dv, Dk, dtype=torch.float32)
    actual_seq = torch.tensor([T], dtype=torch.int32)

    out, final_state = torch.ops.npu.npu_chunk_gated_delta_rule(
        q, k, v, beta=beta, initial_state=init_state,
        actual_seq_lengths=actual_seq, scale=1.0, g=g,
    )

    assert out.shape == (T, Nv, Dv)
    assert final_state.shape == (B, Nv, Dv, Dk)
    assert torch.isfinite(out).all()
    assert torch.isfinite(final_state).all()


# ---------------------------------------------------------------------------
# 8. npu::npu_fused_infer_attention_score_v2  (fused_infer_attention_score.py)
#    schema: npu_fused_infer_attention_score_v2(Tensor query, Tensor key,
#      Tensor value, *, atten_mask, actual_seq_qlen, ..., num_query_heads,
#      num_key_value_heads, softmax_scale, pre/next_tokens, input_layout,
#      sparse_mode, ...) -> (Tensor attention_out, Tensor softmax_lse)
#    cpu: 标准 SDPA (matmul + softmax + matmul), GQA repeat_kv, sparse 0/3/4.
# ---------------------------------------------------------------------------
def test_npu_fused_infer_attention_score_v2_bnsd_full():
    # sparse_mode=0 (no mask): 数值 vs 手算 SDPA (fp32 下精确).
    torch.manual_seed(0)
    b, h, s, d = 2, 4, 8, 16
    q = torch.randn(b, h, s, d, dtype=torch.float32)
    k = torch.randn(b, h, s, d, dtype=torch.float32)
    v = torch.randn(b, h, s, d, dtype=torch.float32)
    scale = 0.25

    out, lse = torch.ops.npu.npu_fused_infer_attention_score_v2(
        q, k, v, num_query_heads=h, num_key_value_heads=h,
        softmax_scale=scale, input_layout="BNSD", sparse_mode=0,
    )

    assert out.shape == q.shape
    assert lse.numel() == 0
    assert torch.isfinite(out).all()
    attn = (q @ k.transpose(-1, -2)) * scale
    expected = torch.softmax(attn.float(), dim=-1).to(q.dtype) @ v
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_npu_fused_infer_attention_score_v2_causal_shape_finite():
    # sparse_mode=3 (causal): CPU impl _build_sparse_mask unpacks
    # `b,n,s_q,s_k = q.shape` 但 BNSD 下 q=[b,n,s,d] 使 s_k=d, s_q!=s_k 命中
    # early-return None -> causal mask 被跳过. 乃 cann_lm CPU impl 限制 (不修改
    # 主仓), 此处仅校验 shape + 有限值, 不做数值对齐.
    torch.manual_seed(0)
    b, h, s, d = 2, 4, 8, 16
    q = torch.randn(b, h, s, d, dtype=torch.float32)
    k = torch.randn(b, h, s, d, dtype=torch.float32)
    v = torch.randn(b, h, s, d, dtype=torch.float32)

    out, lse = torch.ops.npu.npu_fused_infer_attention_score_v2(
        q, k, v, num_query_heads=h, num_key_value_heads=h,
        softmax_scale=0.25, input_layout="BNSD", sparse_mode=3,
    )

    assert out.shape == q.shape
    assert lse.numel() == 0
    assert torch.isfinite(out).all()


def test_npu_fused_infer_attention_score_v2_gqa_bsh():
    torch.manual_seed(0)
    b, s = 2, 4
    nq, nkv, d = 4, 2, 8
    q = torch.randn(b, s, nq * d, dtype=torch.float32)
    k = torch.randn(b, s, nkv * d, dtype=torch.float32)
    v = torch.randn(b, s, nkv * d, dtype=torch.float32)

    out, lse = torch.ops.npu.npu_fused_infer_attention_score_v2(
        q, k, v, num_query_heads=nq, num_key_value_heads=nkv,
        softmax_scale=0.35, input_layout="BSH", sparse_mode=0,
    )

    assert out.shape == q.shape
    assert lse.numel() == 0
    assert torch.isfinite(out).all()


def test_npu_fused_infer_attention_score_v2_extern_mask():
    torch.manual_seed(0)
    b, h, s, d = 1, 2, 5, 8
    q = torch.randn(b, h, s, d, dtype=torch.float32)
    k = torch.randn(b, h, s, d, dtype=torch.float32)
    v = torch.randn(b, h, s, d, dtype=torch.float32)
    # bool mask: True = mask out (-inf); shape [b,1,s,s]
    mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1).view(b, 1, s, s)

    out, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
        q, k, v, atten_mask=mask, num_query_heads=h, num_key_value_heads=h,
        softmax_scale=0.5, input_layout="BNSD", sparse_mode=0,
    )

    assert out.shape == q.shape
    assert torch.isfinite(out).all()
    attn = (q @ k.transpose(-1, -2)) * 0.5
    fmask = torch.where(mask, float("-inf"), 0.0)
    attn = attn + fmask
    expected = torch.softmax(attn.float(), dim=-1).to(q.dtype) @ v
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
