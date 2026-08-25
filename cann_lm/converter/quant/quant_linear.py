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

"""Export-path QuantLinear: forward emits npu::npu_quant_matmul nodes.

Replaces nn.Linear so torch.export traces ``torch.ops.npu.npu_quant_matmul``
(instead of aten.mm/MatMul), mapped by ``_quant_matmul_converter.py`` to
ge.QuantBatchMatmulV4.

Stores materialized int4 weight (nibble-packed int8 ``[k//2, n]``) + per-group
fp32 scale ``[k//g, n]`` as buffers (graph constants → weights_*.bin).

Convention (W4A8 s8s4 对称, offset=0):
  - weight_q ``[k//2, n]`` / weight_scale ``[k//g, n]`` **pre-transposed** (swap.py
    物化后 transpose [n,k//2]->[k//2,n]): ``npu::npu_quant_matmul`` meta 取
    ``n = x2.size(-1)`` 且 GE converter 硬编码 ``transpose_x2=False``, 故 x2 必须
    预转置使 ``size(-1) = n`` (out_features); ``transpose_x2=False``。
  - activation: ``npu_dynamic_quant`` → int8 x1 + per-token fp32 x1_scale。
  - bias optional ``[n]`` in output dtype (None when source Linear has no bias;
    no dummy-zeros workaround — the op's bias is genuinely optional).
"""

import torch
from torch import nn

from cann_lm.models.fused_ops import ops as _fused_ops  # noqa: F401  (registers npu_quant_matmul + npu_dynamic_quant)

_OUT_DTYPE_CODE = {torch.float16: 5, torch.bfloat16: 7, torch.float32: 6}


class QuantLinear(nn.Module):
    """nn.Linear replacement whose forward calls npu::npu_quant_matmul.

    Args:
        weight_q: int8 ``[k//2, n]`` — int4 nibble-packed (low nibble=偶 k 索引,
            two's-complement 4-bit signed), **pre-transposed** by swap.py from
            materialize's ``[n, k//2]`` natural orientation (npu_quant_matmul meta
            takes ``n = x2.size(-1)``).
        weight_scale: fp32 ``[k//group, n]`` (per K-group × per output-channel,
            pre-transposed from GPTQ's ``[n, k//group]``).
        group_size: K-axis group size g (``k % g == 0``).
        bias: optional bias ``[n]`` (fp16/bf16/fp32); None if the source Linear had bias=None.
        act_quant: must be True (W4A8 s8s4). False (W4A16) is not supported by this op.
    """

    def __init__(self, weight_q: torch.Tensor, weight_scale: torch.Tensor,
                 group_size: int, bias: torch.Tensor | None = None,
                 act_quant: bool = True):
        super().__init__()
        if not act_quant:
            raise NotImplementedError(
                "QuantLinear (npu::npu_quant_matmul) supports W4A8 s8s4 only; "
                "act_quant=False (W4A16) not supported. Use the legacy path or add a W4A16 op."
            )
        self.group_size = group_size
        self.act_quant = act_quant
        self.register_buffer("weight_q", weight_q.to(torch.int8))       # [k//2, n] nibble-packed (pre-transposed)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))  # [k//g, n]; 落盘转 int64 见 quant_design.md
        if bias is not None:
            self.register_buffer("bias", bias.to(weight_scale.dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_scale = torch.ops.npu.npu_dynamic_quant(x)  # int8 [..., m, k] + per-token fp32 [..., m]
        out = torch.ops.npu.npu_quant_matmul(
            x_q, self.weight_q, self.weight_scale,
            pertoken_scale=x_scale,               # per-token act scale from npu_dynamic_quant
            bias=self.bias,                       # None when source Linear has no bias (no dummy workaround)
            group_sizes=[1, 1, self.group_size],  # V4 3-element [m,n,k]; K-group at index [2]
            output_dtype=_OUT_DTYPE_CODE.get(x.dtype, 5),
            transpose_x1=False,
            transpose_x2=False,                   # x2 pre-transposed to [k//2, n] in swap; meta takes N=x2.size(-1)
        )
        return out
