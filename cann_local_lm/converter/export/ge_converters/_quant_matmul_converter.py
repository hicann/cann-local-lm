# Copyright (c) 2023-2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Adapted from torchair (Copyright (c) 2023, Huawei Technologies Co., Ltd, BSD License):
#   - torchair/python/torchair/_ge_concrete_graph/ge_converter/custom/quant_matmul.py
# and torch_npu (Copyright (c) Huawei Technologies Co., Ltd, BSD License).
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

"""cann_local_lm::quant_matmul GE converter (V4-only, no torch_npu dependency).

Simplified from torchair npu_quant_matmul: drops the mxfp4/a8w4/int4 packed-
layout branches (which require torch_npu dtype constants like
float4_e2m1fn_x2 / float8_e8m0fnu that have no torch equivalent). Supports
basic quantized matmul via ge.QuantBatchMatmulV4 with per-tensor
scale/offset/bias and per-group group_sizes. Registered lazily by
cann_local_lm.converter.export.ge_converter.register_ge_converters().
"""

from typing import List, Optional

import torch
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
from torchair._ge_concrete_graph.ge_converter.converter_utils import (
    GROUP_SIZE_MAX_VALUE,
)
from torchair.ge import attr
from torchair.ge._ge_graph import (
    DataType,
    ProtoDataType,
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
    """cann_local_lm::quant_matmul -> ge.QuantBatchMatmulV4 (basic path, no torch_npu)."""
    del transpose_x1, transpose_x2, meta_outputs
    if output_dtype is None:
        output_dtype = 1
    dtype = torch_dtype_value_to_ge_type(output_dtype)
    # a8w4: x1=FP8 + x2=FP4/FP32; s8s4: x1=int8 + x2=int4 (nibble-packed int8). 两者 per-group 只在 K 维 -> group_size=group_k
    # (绕过 torch_npu.float4_e2m1fn_x2, 用 x1/x2.dtype 判断; Bitcast 前判断, x2 此时仍是 packed int8=DT_INT8)
    is_s8s4 = x1.dtype == DataType.DT_INT8 and x2.dtype == DataType.DT_INT8

    if x1_dtype is not None and x1 is not None:
        x1 = ge.Bitcast(x1, type=torch_dtype_value_to_ge_type(x1_dtype))
        x1.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
    if x2_dtype is not None and x2 is not None:
        x2 = ge.Bitcast(x2, type=torch_dtype_value_to_ge_type(x2_dtype))
        x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
    if is_s8s4:
        # x2 是 int8 nibble-packed [N, K/2] (2 int4/int8), ND 格式.
        # 导出侧只做类型改写: Bitcast (int8→int4, [N, K/2]→[N, K] 字节不变), x2 以
        # int4 ND 声明进图. 不插 TransData — 实际 ND→FRACTAL_Z 权重重排由编译期
        # host 侧自定义算子 (op_cpu 注册 Bitcast/TransData HostCpu kernel,
        # ConstantFolding 折叠) 与 CustomPass 完成.
        x2_in_dims = list(x2.desc.shape.dim) if x2.desc.shape.dim else []
        n_val = int(x2_in_dims[0]) if len(x2_in_dims) >= 1 else None

        # Bitcast: [N, K/2] int8 → [N, K] int4.
        # HostCpu 注册 (op_cpu/bitcast_cpu.py) 的 Bitcast IR 输出名是 "output0"
        # (其注册 schema 硬编码 output{index}), epair loader 按注册表 IR 校验
        # 输出端口名; 原生 ge.Bitcast 输出名是 "y", 端口名不匹配会在装载期被拒
        # ("Output y is not compatible with IR output output0"). 因此必须用
        # ge_op + IrDef 显式对齐 HostCpu 注册的 IR 定义.
        x2 = ge_op(
            op_type="Bitcast",
            inputs={"x": x2},
            attrs={"type": attr.DataType(DataType.DT_INT4)},
            outputs=["output0"],
            ir=IrDef("Bitcast"),
        )
        x2.desc.dtype = ProtoDataType.DT_INT4
        x2.desc.layout = "ND"
        if x2_in_dims:
            del x2.desc.shape.dim[:]
            bc_dims = list(x2_in_dims)
            bc_dims[-1] = bc_dims[-1] * 2  # K/2 → K (int4 element count)
            x2.desc.shape.dim.extend(bc_dims)
    if pertoken_scale is not None and pertoken_scale_dtype is not None:
        pertoken_scale = ge.Bitcast(
            pertoken_scale, type=torch_dtype_value_to_ge_type(pertoken_scale_dtype)
        )
        pertoken_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(
            pertoken_scale_dtype
        )
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
    # 新流程: 权重保持 ND ([N, K/2] int8 → Bitcast → [N, K] int4 ND), 导出不预排
    # FRACTAL_Z; 编译期由 host 侧自定义算子 (op_cpu, ASCEND_CUSTOM_OPP_PATH 注册) 与
    # CustomPass 将 x2 落到 int4 FRACTAL_Z.
    # Use ge.QuantBatchMatmulV4 (full IR with all inputs incl None placeholders) so the
    # V4→V3 TBE fusion correctly maps x2_scale to V3's required scale input.
    if bias is None:
        # CANN QuantBatchMatmulV4 InferShape (InferNDimWithBias) does NOT handle
        # bias=nullptr — it tries to infer bias shape from x1's M dim and compares
        # to N, failing with "dimensions a(M) and b(N) must be same".
        # Workaround: create a dummy bias (zeros [N], float32) so InferShape passes.
        # The bias value (0) does not affect the matmul result (out = x1@x2*scale + 0).
        # N 是权重的行数; is_s8s4 时 n_val 已从 Bitcast 前的 x2 shape 提取,
        # 否则从 x2 最后一维取 (旧路径兼容).
        bias_n = n_val if is_s8s4 else (x2.shape[-1] if hasattr(x2, "shape") else None)
        bias = ge.Const([0.0] * int(bias_n), dtype=1) if bias_n else None

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
    # arch31 s8s4 tiling requires x1=int8 ND, x2=int4 FRACTAL_Z.
    # 导出侧 x2 以 int4 ND 声明; 落 FRACTAL_Z 由编译期 host 侧算子 + CustomPass 负责.
    # input_layout_info (torchair convention) 同步声明为 ND (node 生成阶段 x2 仍是 ND).
    if is_s8s4:
        node = out.node
        node.attr["input_layout_info"].list.i.extend([0, 1])
        node.attr["input_layout_info"].list.s.extend([b"ND", b"ND"])
    out.desc.dtype = torch_dtype_value_to_ge_proto_type(output_dtype)
    return out
