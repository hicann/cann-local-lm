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

"""Export-path QuantLinear (方案 A): forward emits cann_lm::quant_matmul_s8s4 nodes.

Replaces nn.Linear so torch.export traces ``torch.ops.cann_lm.quant_matmul_s8s4``
(instead of aten.mm/MatMul), mapped by
``_quant_matmul_s8s4_converter.py`` to the 方案 A custom GE op (待 kernel).

Stores materialized int4 weight (nibble-packed int8 ``[n, k//2]``) + per-group
fp32 scale ``[n, k//g]`` as buffers (graph constants → weights_*.bin).

Convention (方案 A, s8s4 对称, offset=0):
  - weight_q / weight_scale in **natural** nn.Linear orientation ``[n, k]`` /
    ``[n, k//g]`` (NO pre-transpose); ``transpose_x2=True`` (kernel transposes
    internally for the cube).
  - activation: ``npu_dynamic_quant`` → int8 x1 + per-token fp32 x1_scale.
  - bias optional ``[n]`` in output dtype (None when source Linear has no bias;
    no dummy-zeros workaround — the 方案 A op's bias is genuinely optional).
"""

import torch
from torch import nn

from cann_lm.models.fused_ops import ops as _fused_ops  # noqa: F401  (registers quant_matmul_s8s4 + npu_dynamic_quant)

_OUT_DTYPE_CODE = {torch.float16: 5, torch.bfloat16: 7, torch.float32: 6}


class QuantLinear(nn.Module):
    """nn.Linear replacement whose forward calls cann_lm::quant_matmul_s8s4 (方案 A).

    Args:
        weight_q: int8 ``[n, k//2]`` — int4 nibble-packed (low nibble=偶 k 索引,
            two's-complement 4-bit signed). From ``materialize_int4_with_scale(..., packed=True)``.
        weight_scale: fp32 ``[n, k//group]`` (per output-channel × per K-group, from GPTQ).
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
                "QuantLinear (cann_lm::quant_matmul_s8s4) supports W4A8 s8s4 only; "
                "act_quant=False (W4A16) not supported. Use the legacy path or add a W4A16 op."
            )
        self.group_size = group_size
        self.act_quant = act_quant
        self.register_buffer("weight_q", weight_q.to(torch.int8))       # [n, k//2] nibble-packed
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))  # [n, k//g]
        if bias is not None:
            self.register_buffer("bias", bias.to(weight_scale.dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_scale = torch.ops.npu.npu_dynamic_quant(x)  # int8 [..., m, k] + per-token fp32 [..., m]
        out = torch.ops.cann_lm.quant_matmul_s8s4(
            x_q, x_scale, self.weight_q, self.weight_scale,
            bias=self.bias,                       # None when source Linear has no bias (no dummy workaround)
            group_size=self.group_size,           # single int attr (NOT V4's [m,n,k] 48-bit pack)
            output_dtype=_OUT_DTYPE_CODE.get(x.dtype, 5),
            transpose_x1=False,
            transpose_x2=True,                    # x2 given [n, k//2] natural; kernel transposes
        )
        return out
