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

"""cann_lm::quant_matmul GE converter (V4-only, no torch_npu dependency).

Simplified from torchair npu_quant_matmul: drops the mxfp4/a8w4/int4 packed-
layout branches (which require torch_npu dtype constants like
float4_e2m1fn_x2 / float8_e8m0fnu that have no torch equivalent). Supports
basic quantized matmul via ge.QuantBatchMatmulV4 with per-tensor
scale/offset/bias and per-group group_sizes. Registered lazily by
cann_lm.converter.export.ge_converter.register_ge_converters().
"""

from typing import List, Optional

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.ge_converter.converter_utils import GROUP_SIZE_MAX_VALUE
from torchair.ge._ge_graph import (
    DataType,
    Tensor,
    TensorSpec,
    torch_dtype_value_to_ge_proto_type,
    torch_dtype_value_to_ge_type,
)


def convert_quant_matmul(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    *,
    offset: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    output_dtype: int = None,
    x1_dtype: int = None,
    x2_dtype: int = None,
    pertoken_scale_dtype: int = None,
    scale_dtype: int = None,
    group_sizes: Optional[List[int]] = None,
    y_scale: Optional[Tensor] = None,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    meta_outputs: TensorSpec = None,
):
    """cann_lm::quant_matmul -> ge.QuantBatchMatmulV4 (basic path, no torch_npu)."""
    del transpose_x1, transpose_x2, meta_outputs
    if output_dtype is None:
        output_dtype = 1
    dtype = torch_dtype_value_to_ge_type(output_dtype)
    # a8w4: x1=FP8 + x2=FP4/FP32; s8s4: x1=int8 + x2=int4 (nibble-packed int8). 两者 per-group 只在 K 维 -> group_size=group_k
    # (绕过 torch_npu.float4_e2m1fn_x2, 用 x1/x2.dtype 判断; Bitcast 前判断, x2 此时仍是 packed int8=DT_INT8)
    is_a8w4 = x1.dtype == DataType.DT_FLOAT8_E4M3FN and (
        x2.dtype in (DataType.DT_FLOAT4_E2M1, DataType.DT_FLOAT)
    )
    is_s8s4 = x1.dtype == DataType.DT_INT8 and x2.dtype == DataType.DT_INT8

    if x1_dtype is not None and x1 is not None:
        x1 = ge.Bitcast(x1, type=torch_dtype_value_to_ge_type(x1_dtype))
        x1.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
    if x2_dtype is not None and x2 is not None:
        x2 = ge.Bitcast(x2, type=torch_dtype_value_to_ge_type(x2_dtype))
        x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
    if pertoken_scale is not None and pertoken_scale_dtype is not None:
        pertoken_scale = ge.Bitcast(
            pertoken_scale, type=torch_dtype_value_to_ge_type(pertoken_scale_dtype)
        )
        pertoken_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(pertoken_scale_dtype)
    if scale_dtype is not None:
        # scale_dtype 直接作 CANN DataType 值 (如 DT_UINT64=10); torch 无 uint64 ScalarType,
        # torch_dtype_value_to_ge_type 会把 10 误解析为 complex128(17), 故直接用值.
        scale = ge.Bitcast(scale, type=scale_dtype)
        scale.desc.dtype = scale_dtype

    group_size = 0
    if group_sizes is not None:
        if isinstance(group_sizes, List):
            if not group_sizes:
                raise RuntimeError("group_sizes is empty")
            group_k = group_sizes[-1]  # 取末元素 (兼容 [1,1,g] / [g] 等任意长度)
        else:
            group_k = int(group_sizes)
        group_max = GROUP_SIZE_MAX_VALUE
        if group_k > group_max or group_k < 0:
            raise RuntimeError(f"group_k {group_k} out of [0, {group_max}]")
        group_size = group_k  # 写死 group_k (per-K-group, 不编码 m/n/k)

    # Use ge.QuantBatchMatmulV4 (full IR with all inputs incl None placeholders) so the
    # V4→V3 TBE fusion correctly maps x2_scale to V3's required scale input.
    # The binary must have QuantBatchMatmulV4 registered (reg_register.h) for the
    # C++ parser to create the op node without heap corruption.
    if bias is None:
        # CANN QuantBatchMatmulV4 InferShape (InferNDimWithBias) does NOT handle
        # bias=nullptr — it tries to infer bias shape from x1's M dim and compares
        # to N, failing with "dimensions a(M) and b(N) must be same".
        # Workaround: create a dummy bias (zeros [N], float32) so InferShape passes.
        # The bias value (0) does not affect the matmul result (out = x1@x2*scale + 0).
        n = x2.shape[-1] if hasattr(x2, 'shape') else None
        bias = ge.Const([0.0] * int(n), dtype=1) if n else None

    out = ge.QuantBatchMatmulV4(
        x1,
        x2,
        bias=bias,
        x1_scale=pertoken_scale,
        x2_scale=scale,
        y_scale=y_scale,
        x1_offset=None,
        x2_offset=offset,
        y_offset=None,
        x2_table=None,
        dtype=dtype,
        transpose_x1=False,
        transpose_x2=False,
        group_size=group_size,
    )
    out.desc.dtype = torch_dtype_value_to_ge_proto_type(output_dtype)
    return out
