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

"""cann_lm::quant_matmul - quantized matmul op (schema + meta, export-only).

遗留问题: a8w4 量化格式未支持。
当前 converter (``_quant_matmul_converter.py``) 为去除 torch_npu 依赖,去掉了
a8w4 分支 (x1=FP8_E4M3FN + x2=FP4/FP32-pack + per-group ``y_scale`` +
``group_size``)。该分支依赖 ``torch_npu.float4_e2m1fn_x2`` dtype 常量 (torch
无等价物)。恢复 a8w4 支持需重新引入 torch_npu 依赖或建立 dtype 常量映射。
"""

import torch

from ._library import NPU_LIB, _HAS_TORCH_NPU

if NPU_LIB is not None:
    NPU_LIB.define(
        "npu_quant_matmul("
        "Tensor x1, Tensor x2, Tensor scale, *, "
        "Tensor? offset=None, Tensor? pertoken_scale=None, Tensor? bias=None, "
        "int? output_dtype=None, int? x1_dtype=None, int? x2_dtype=None, "
        "int? pertoken_scale_dtype=None, int? scale_dtype=None, "
        "int[]? group_sizes=None, Tensor? y_scale=None, "
        "bool transpose_x1=False, bool transpose_x2=False) -> (Tensor y)"
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
    @torch.library.register_fake("npu::npu_quant_matmul")
    def _quant_matmul_meta(
        x1: torch.Tensor,
        x2: torch.Tensor,
        scale: torch.Tensor,
        *,
        offset: torch.Tensor | None = None,
        pertoken_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        output_dtype: int | None = None,
        x1_dtype: int | None = None,
        x2_dtype: int | None = None,
        pertoken_scale_dtype: int | None = None,
        scale_dtype: int | None = None,
        group_sizes: list[int] | None = None,
        y_scale: torch.Tensor | None = None,
        transpose_x1: bool = False,
        transpose_x2: bool = False,
    ) -> torch.Tensor:
        del (
            offset, pertoken_scale, bias, x1_dtype, x2_dtype, pertoken_scale_dtype,
            scale_dtype, group_sizes, y_scale, transpose_x1, transpose_x2, scale,
        )
        out_dtype = _SCALAR_TYPE_TO_TORCH_DTYPE.get(output_dtype, torch.int8)
        m = x1.size(-2)
        n = x2.size(-1)
        batch = list(torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
        return x1.new_empty(batch + [m, n], dtype=out_dtype)
