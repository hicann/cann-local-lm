"""手算微夹具 (bedrock) —— 人用算术算出的期望值，不依赖任何量化代码。

每个 case 给出输入 + 手算 expected + 推导。ref_w4a8.self_check() 会拿这些去对
reference 实现。这是整个验证可信度的根：最底层是人的算术。

spec 约定（ref_w4a8 实现 + 本夹具共同编码）：
  权重 int4 对称 per-group: q_w = clamp(round(w/s), -7, 7), offset=0, dequant = q_w*s
  激活 int8 对称 per-token:  scale = absmax/127, x_q = clamp(round(x/scale), -128, 127)
  matmul: out = (x_q*scale) @ (q_w*s_expanded).T + bias
  round = torch.round (half-to-even, 银行家舍入) —— 与 converter materialize/dynamic_quant 一致
"""

import torch

# (name, kind, inputs, expected, tol)
# kind: "weight" -> 验 q_w; "act" -> 验 (x_q, x_scale); "matmul" -> 验 out
FIXTURES = [
    {
        "name": "W1_weight_basic",
        "kind": "weight",
        "w": [[2.0, -3.0, 1.5, 0.7]],
        "scale": [[1.0]],
        "group_size": 4,
        "expected_qw": [[2, -3, 2, 1]],
        "derive": "round(2)=2, round(-3)=-3, round(1.5)=2(half-to-even), round(0.7)=1; clamp[-7,7] unchanged",
    },
    {
        "name": "W2_weight_clamp",
        "kind": "weight",
        "w": [[10.0, -10.0, 0.3, 0.4]],
        "scale": [[1.0]],
        "group_size": 4,
        "expected_qw": [[7, -8, 0, 0]],
        "derive": "round->[10,-10,0,0]; clamp[-8,7]->[7,-8,0,0] (-10 clamp 到 -8, 保住 -8 不丢)",
    },
    {
        "name": "W2b_neg8_kept",
        "kind": "weight",
        "w": [[-8.0, -9.0, 8.0, 7.0]],
        "scale": [[1.0]],
        "group_size": 4,
        "expected_qw": [[-8, -8, 7, 7]],
        "derive": "round->[-8,-9,8,7]; clamp[-8,7]->[-8,-8,7,7] (-8 原样保留, -9 clamp -8, 8 clamp 7)",
    },
    {
        "name": "W3_round_half_to_even",
        "kind": "weight",
        "w": [[0.5, 1.5, 2.5, 3.5]],
        "scale": [[1.0]],
        "group_size": 4,
        "expected_qw": [[0, 2, 2, 4]],
        "derive": "half-to-even: 0.5->0, 1.5->2, 2.5->2, 3.5->4",
    },
    {
        "name": "A1_dynamic_quant",
        "kind": "act",
        "x": [[1.0, 2.0, -1.0, 0.5]],
        "expected_xq": [[64, 127, -64, 32]],
        "expected_scale": 2.0 / 127.0,
        "tol": 1e-6,
        "derive": "absmax=2, scale=2/127; x/scale=[63.5,127,-63.5,31.75]; round: 63.5->64,127->127,-63.5->64(even neg->-64),31.75->32; clamp unchanged",
    },
    {
        "name": "M1_full_w4a8_matmul",
        "kind": "matmul",
        "w": [[2.0, -3.0, 1.5, 0.7]],
        "scale": [[1.0]],
        "group_size": 4,
        "x": [[1.0, 2.0, -1.0, 0.5]],
        "bias": None,
        "expected_out": -698.0 / 127.0,
        "tol": 1e-4,
        "derive": "q_w=[2,-3,2,1] dequant_w=[2,-3,2,1]; x_q=[64,127,-64,32] x_scale=2/127; "
        "dequant_x=[128/127,2,-128/127,64/127]; out=256/127-6-256/127+64/127 = -6+64/127 = -698/127 ~= -5.496",
    },
]


def to_tensor(x):
    return (
        torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
    )
