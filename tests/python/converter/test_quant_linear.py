"""quant_swap/quant_linear.py 单元测试 — QuantLinear __init__ (buffer 注册 + act_quant 校验).

forward 不在 UT 范围: QuantLinear 存 nibble-packed weight_q [k//2,n] (GE export 路径,
GE 内部解 nibble), 而 CPU impl _npu_quant_matmul_cpu 期望 unpacked [K,N]
("int4 stored unpacked", eager 回归用) — 格式不匹配, forward 走 torch.export→GE
(e2e, run_e2e_pipeline.sh 验证), 不可 CPU 单测. 仅测 __init__ 可测面.
"""

import pytest
import torch

from cann_local_lm.converter.export.quant_swap.quant_linear import (
    QuantLinear,
    _OUT_DTYPE_CODE,
)


def test_out_dtype_code_mapping():
    assert _OUT_DTYPE_CODE[torch.float16] == 5
    assert _OUT_DTYPE_CODE[torch.bfloat16] == 7
    assert _OUT_DTYPE_CODE[torch.float32] == 6


def test_init_registers_buffers_no_bias():
    k, n, g = 8, 4, 4
    wq = torch.randint(-8, 7, (k // 2, n), dtype=torch.int8)  # nibble-packed [k//2, n]
    ws = torch.randn(k // g, n, dtype=torch.float32) + 0.1
    ql = QuantLinear(wq, ws, group_size=g)
    assert ql.weight_q.dtype == torch.int8 and ql.weight_q.shape == (k // 2, n)
    assert ql.weight_scale.dtype == torch.float32 and ql.weight_scale.shape == (
        k // g,
        n,
    )
    assert ql.bias is None
    assert ql.group_size == g and ql.act_quant is True


def test_init_with_bias_registers_buffer():
    k, n, g = 8, 4, 4
    wq = torch.randint(-8, 7, (k // 2, n), dtype=torch.int8)
    ws = torch.randn(k // g, n, dtype=torch.float32) + 0.1
    bias = torch.randn(n, dtype=torch.float32)
    ql = QuantLinear(wq, ws, group_size=g, bias=bias)
    assert (
        ql.bias is not None and ql.bias.shape == (n,) and ql.bias.dtype == torch.float32
    )


def test_init_act_quant_false_raises():
    k, n, g = 8, 4, 4
    wq = torch.zeros(k // 2, n, dtype=torch.int8)
    ws = torch.ones(k // g, n, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="W4A8 s8s4 only"):
        QuantLinear(wq, ws, group_size=g, act_quant=False)


def test_init_dtype_preserved_for_bf16_bias():
    # bias 按 weight_scale.dtype 注册 (bf16 场景)
    k, n, g = 8, 4, 4
    wq = torch.randint(-8, 7, (k // 2, n), dtype=torch.int8)
    ws = torch.randn(k // g, n, dtype=torch.float32) + 0.1
    bias = torch.randn(n, dtype=torch.bfloat16)
    ql = QuantLinear(wq, ws, group_size=g, bias=bias)
    assert ql.bias.dtype == torch.float32  # bias.to(weight_scale.dtype=fp32)
