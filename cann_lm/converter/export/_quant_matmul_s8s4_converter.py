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

"""cann_lm::quant_matmul_s8s4 GE converter (方案 A, 骨架 — 待算子实现方补 kernel).

Maps ``torch.ops.cann_lm.quant_matmul_s8s4`` -> a GE op node of type
``QuantMatmulS8S4`` (manual op-add, same pattern as the ChunkGatedDeltaRule
converter in ``ge_converter.py``). The NPU kernel (AscendC) + ``REG_OP`` proto
in ``reg_register.h`` + 预编译 ``.o`` are provided by the算子实现方; until then
the ep_json carries the ``QuantMatmulS8S4`` node but OM compile reports
``UNMAPPED_OPERATOR`` (resolved once the REG_OP is registered + the .o is in
place). This file only emits the GE graph node; it does NOT implement the kernel.
"""

from typing import Any


def convert_quant_matmul_s8s4(
    x1: Any,
    x1_scale: Any,
    x2: Any,
    x2_scale: Any,
    *,
    bias: Any = None,
    group_size: int = 256,
    output_dtype: int = 5,
    transpose_x1: bool = False,
    transpose_x2: bool = True,
    meta_outputs: Any = None,
) -> Any:
    """cann_lm::quant_matmul_s8s4 -> ge op node "QuantMatmulS8S4" (方案 A 骨架)."""
    from torchair.ge._ge_graph import (
        get_default_ge_graph,
        next_unique_name,
        Tensor,
        get_invalid_desc,
        torch_dtype_value_to_ge_proto_type,
    )

    op = get_default_ge_graph().op.add()
    op.type = "QuantMatmulS8S4"
    op.name = next_unique_name(None, "QuantMatmulS8S4")

    # inputs: x1 (int8 act), x1_scale (fp32 pertoken), x2 (int4 packed), x2_scale (fp32), bias?
    input_list = [
        (x1, "x1"),
        (x1_scale, "x1_scale"),
        (x2, "x2"),
        (x2_scale, "x2_scale"),
        (bias, "bias"),
    ]
    for item_t, item_n in input_list:
        if item_t is not None:
            op.input.append(item_t.tensor)
            op.input_desc.add().CopyFrom(item_t.desc)
            op.input_desc[-1].name = item_n
        else:
            op.input.append("")
            op.input_desc.add().CopyFrom(get_invalid_desc())
            op.input_desc[-1].name = item_n

    # attrs: group_size (K-axis), output_dtype (1=fp16/27=bf16 scalar code),
    # transpose_x1/transpose_x2 (x2 given [n,k] natural; transpose_x2=True).
    op.attr["group_size"].i = int(group_size)
    op.attr["output_dtype"].i = int(output_dtype)
    op.attr["transpose_x1"].b = bool(transpose_x1)
    op.attr["transpose_x2"].b = bool(transpose_x2)

    # output y (single): shape from meta_outputs, dtype from output_dtype code.
    op.output_desc.add().name = "y"
    out = Tensor(op, 0)
    mo = meta_outputs[0] if isinstance(meta_outputs, (list, tuple)) else meta_outputs
    if mo is not None:
        del out.desc.shape.dim[:]
        out.desc.shape.dim.extend(list(getattr(mo, "size", []) or []))
    out.desc.dtype = torch_dtype_value_to_ge_proto_type(output_dtype)
    return out
