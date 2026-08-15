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

"""cann_lm::quant_matmul_s8s4 — 方案 A 自研 W4A8 量化 matmul (schema + meta + cpu).

绕开 CANN V4 的 uint64 deqScale/FP19 + V4→V3 融合降级: 用 **fp32 per-group
scale** 做 ``q*scale`` 反量化, int8 cube 累加。算子签名见 ``quant_design.md``
"方案 A — 自研 quant_matmul_s8s4 算子规格"。

输入 (W4A8 s8s4 对称, offset=0):
  x1       int8  [..., m, k]  (per-token 量化, 来自 npu_dynamic_quant)
  x1_scale fp32 [..., m]      (pertoken, 来自 npu_dynamic_quant)
  x2       int4 nibble-packed int8 [n, k//2]  (nn.Linear 权重 [n,k] 自然朝向, 低 nibble=偶 k 索引)
  x2_scale fp32 [n, k//group] (per-channel × per-group, 来自 GPTQ)
输出:
  y        fp16/bf16 [..., m, n]

NPU kernel (AscendC) + GE ``REG_OP(QuantMatmulS8S4)`` + 预编译 .o 由算子实现方提供;
本文件仅 Python schema + register_fake (export 用) + cpu eager impl (数值回归)。
GE 映射见 ``cann_lm/converter/export/_quant_matmul_s8s4_converter.py`` (骨架, 待 kernel)。
"""

import torch

from ._library import CANN_LM_LIB, _HAS_TORCH_NPU, register_cpu_impl

if CANN_LM_LIB is not None:
    CANN_LM_LIB.define(
        "quant_matmul_s8s4("
        "Tensor x1, Tensor x1_scale, Tensor x2, Tensor x2_scale, *, "
        "Tensor? bias=None, int group_size, int output_dtype=5, "
        "bool transpose_x1=False, bool transpose_x2=True) -> Tensor y"
    )

_SCALAR_TYPE_TO_TORCH_DTYPE = {
    0: torch.uint8,
    1: torch.int8,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.bfloat16,
}


if not _HAS_TORCH_NPU:
    @torch.library.register_fake("cann_lm::quant_matmul_s8s4")
    def _quant_matmul_s8s4_meta(
        x1: torch.Tensor,
        x1_scale: torch.Tensor,
        x2: torch.Tensor,
        x2_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        group_size: int = 256,
        output_dtype: int = 5,
        transpose_x1: bool = False,
        transpose_x2: bool = True,
    ) -> torch.Tensor:
        del x1_scale, x2_scale, bias, group_size, transpose_x1
        out_dt = _SCALAR_TYPE_TO_TORCH_DTYPE.get(output_dtype, torch.float16)
        m = x1.size(-2)
        # x2 physical [n, k//2]; logical [n, k]; transpose_x2=True -> y=[...,m,n], n=x2.size(0)
        n = x2.size(0) if transpose_x2 else x2.size(-1)
        batch = list(torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
        return x1.new_empty(batch + [m, n], dtype=out_dt)

    @register_cpu_impl("cann_lm::quant_matmul_s8s4")
    def _quant_matmul_s8s4_cpu(
        x1: torch.Tensor,
        x1_scale: torch.Tensor,
        x2: torch.Tensor,
        x2_scale: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        group_size: int = 256,
        output_dtype: int = 5,
        transpose_x1: bool = False,
        transpose_x2: bool = True,
    ) -> torch.Tensor:
        """W4A8 s8s4 CPU eager (numerical regression only, not perf).

        y = (x1 * x1_scale) @ (unpack_int4(x2) * x2_scale)^T  + bias, cast output_dtype.
        int4 nibble-pack convention: x2 [n, k//2] int8, 低 nibble=偶 k, 高 nibble=奇 k,
        signed [-8,7] (nibble>7 -> nibble-16)。对称 (无 offset)。
        """
        del transpose_x1
        out_dt = _SCALAR_TYPE_TO_TORCH_DTYPE.get(output_dtype, torch.float16)
        if not transpose_x2:
            raise NotImplementedError(
                "quant_matmul_s8s4 cpu impl only supports transpose_x2=True "
                "(x2 = [n, k//2] nibble-packed natural orientation); got transpose_x2=False"
            )
        # unpack int4: [n, k//2] int8 -> signed int4 values as int8 [n, k]
        wp = x2.to(torch.int8)
        low = wp & 0x0F
        high = (wp >> 4) & 0x0F
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)
        w = torch.stack((low, high), dim=-1).reshape(low.shape[0], -1)  # [n, k]
        # dequant weight: per-group scale [n, k//g] -> expand to [n, k]
        sc = x2_scale.float()
        if group_size > 1:
            sc = sc.repeat_interleave(group_size, dim=-1)
        w_fp = w.float() * sc  # [n, k]
        # dequant act: int8 + pertoken scale (broadcast over k)
        x1_fp = x1.float() * x1_scale.float().unsqueeze(-1)  # [..., m, k]
        out = x1_fp @ w_fp.transpose(-1, -2)  # [..., m, k] @ [k, n] -> [..., m, n]
        if bias is not None:
            out = out + bias.float()
        return out.to(out_dt)
