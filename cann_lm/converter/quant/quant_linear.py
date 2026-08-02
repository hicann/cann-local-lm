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

"""Export-path QuantLinear: forward emits cann_lm fused npu_quant_matmul nodes.

Replaces nn.Linear so that torch.export traces ``torch.ops.npu.npu_quant_matmul``
nodes (instead of aten.mm/MatMul), which the GE converter maps to
``ge.QuantBatchMatmulV4``. Stores materialized int4 weight (int8 unpack) +
per-group scale as buffers (graph constants, packed into weights_*.bin).

Convention (W4A8 unpack+signed PoC, matches the cpu eager kernel + GE converter
which hardcodes transpose_x2=False): the QuantLinear passes weight pre-transposed
to ``[K=in, N=out]`` and scale to ``[K//group, N]`` with ``transpose_x2=False``.
Activation per-token scale (pertoken_scale) is computed at runtime from x.
"""

import torch
from torch import nn

from cann_lm.models.fused_ops import ops as _fused_ops  # noqa: F401  (registers npu_quant_matmul)

_OUT_DTYPE_CODE = {torch.float16: 5, torch.bfloat16: 7, torch.float32: 6}


class QuantLinear(nn.Module):
    """nn.Linear replacement whose forward calls cann_lm::npu_quant_matmul.

    Args:
        weight_q: int8 tensor ``[out, in]`` (int4 values stored unpack, 1/byte).
        weight_scale: fp32 tensor ``[out, in//group]`` (per output-channel × per K-group).
        group_size: K-axis group size.
        bias: optional bias (fp16/bf16/fp32); None if the source Linear had bias=None.
    """

    def __init__(self, weight_q: torch.Tensor, weight_scale: torch.Tensor,
                 group_size: int, bias: torch.Tensor | None = None,
                 act_quant: bool = True):
        super().__init__()
        self.group_size = group_size
        self.act_quant = act_quant
        self.register_buffer("weight_q", weight_q.to(torch.int8))
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))
        if bias is not None:
            self.register_buffer("bias", bias.to(weight_scale.dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-transpose weight+scale to [K=in, N=out] / [K//group, N]; transpose_x2=False
        # (the GE converter _quant_matmul_converter hardcodes transpose_x2=False)
        w2 = self.weight_q.transpose(-1, -2).contiguous()
        sc = self.weight_scale.transpose(-1, -2).contiguous()
        if self.act_quant:
            # W4A8: npu_dynamic_quant (fused op, GE-convertible to ge.DynamicQuant) produces
            # int8 activation + per-token scale — replaces the aten.abs/amax/clamp/div chain
            # (which had no GE converter and blocked export). x_q (int8) feeds npu_quant_matmul.
            x_q, x_scale = torch.ops.npu.npu_dynamic_quant(x)
            x_in = x_q
        else:
            # W4A16: weight-only, no activation quantization
            x_in = x
            x_scale = None
        # CANN QuantBatchMatmulV4 InferShape (InferNDimWithBias) does NOT handle
        # bias=nullptr — it tries to infer bias shape and fails. Pass a dummy
        # zeros bias [out_features] so InferShape passes. Value 0 = no-op.
        if self.bias is None:
            n_out = self.weight_q.shape[0]  # [out, in] → out
            dummy_bias = torch.zeros(n_out, dtype=x.dtype, device=x.device)
        else:
            dummy_bias = self.bias
        out = torch.ops.npu.npu_quant_matmul(
            x_in, w2, sc,
            pertoken_scale=x_scale,
            bias=dummy_bias,
            output_dtype=_OUT_DTYPE_CODE.get(x.dtype, 5),
            group_sizes=[1, 1, self.group_size],  # K-G mode (per-group, matches 2D scale [out, in//group])
            transpose_x1=False,
            transpose_x2=False,
        )
        if self.bias is not None:
            out = out + self.bias
        return out
