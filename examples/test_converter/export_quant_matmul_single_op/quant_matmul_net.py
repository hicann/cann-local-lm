"""Minimal single-op module: one npu_quant_matmul call (W4A8 baseline).

Tests whether QuantBatchMatmulV4 with the CURRENT converter config
(transpose_x2=False, x2=int8-unpack-int4, no x2_dtype Bitcast) hits the
prebuilt `ND_ND_int8_int4` kernel (tilingKey=24) on OM compile, or fails.

Layout (matches QuantLinear + _quant_matmul_converter baseline):
  weight_q   : int8 [n, k]  (int4 signed values [-8,7] stored unpack, 1/byte)
  weight_scale: fp32 [n, k//group]  (per-channel × per-K-group)
  x1 (act)   : int8 [m, k]  (runtime input, from DynamicQuant)
  pertoken    : fp32 [m]    (runtime input, per-token act scale)

forward pre-transposes weight+scale to [k, n] / [k//group, n] and calls
npu_quant_matmul with transpose_x2=False (what _quant_matmul_converter
hardcodes). out=fp16, group_sizes=[1,1,group_size].
"""

import torch
from torch import nn


class QuantMatmulSingleOp(nn.Module):
    def __init__(self, m: int = 4, k: int = 256, n: int = 8, group_size: int = 256):
        super().__init__()
        self.group_size = group_size
        self.n_out = n
        # int4 signed values stored unpack as int8 (matches materialize_int4 output)
        wq = torch.randint(-8, 7, (n, k), dtype=torch.int8)
        self.register_buffer("weight_q", wq)
        # per-channel × per-K-group scale, fp32
        sc = torch.rand(n, k // group_size, dtype=torch.float32) * 0.01 + 1e-3
        self.register_buffer("weight_scale", sc)

    def forward(self, x1: torch.Tensor, pertoken_scale: torch.Tensor) -> torch.Tensor:
        # pre-transpose to [k, n] / [k//group, n]; transpose_x2=False (converter baseline)
        w2 = self.weight_q.transpose(-1, -2).contiguous()
        sc = self.weight_scale.transpose(-1, -2).contiguous()
        # dummy zeros bias so V4 InferNDimWithBias passes (bias=nullptr not handled)
        dummy_bias = torch.zeros(self.n_out, dtype=torch.float16, device=x1.device)
        return torch.ops.npu.npu_quant_matmul(
            x1, w2, sc,
            pertoken_scale=pertoken_scale,
            bias=dummy_bias,
            output_dtype=5,  # 5 = fp16
            group_sizes=[1, 1, self.group_size],
            transpose_x1=False,
            transpose_x2=False,
        )


def main() -> None:
    m, k, n, gs = 4, 256, 8, 256
    net = QuantMatmulSingleOp(m, k, n, gs).eval()
    x1 = torch.randint(-127, 127, (m, k), dtype=torch.int8)
    pertoken = torch.ones(m, dtype=torch.float32) * 0.01
    with torch.no_grad():
        out = net(x1, pertoken)
    print(f"x1(act) int8 {tuple(x1.shape)}, weight_q int8 {tuple(net.weight_q.shape)}, "
          f"scale fp32 {tuple(net.weight_scale.shape)}")
    print(f"out {tuple(out.shape)} {out.dtype}")
    print(f"params: weight_q + weight_scale (constants), x1 + pertoken_scale (runtime inputs)")


if __name__ == "__main__":
    main()
