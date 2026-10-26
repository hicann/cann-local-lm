"""cann_lm::dynamic_quant GE converter (镜像 torchair npu::npu_dynamic_quant)。

根因: cann_lm::dynamic_quant 原本无 FX→GE converter → torchair auto-callback 跑
get_registered_ir_def("DynamicQuant") → Lite EPAIR check_violations 拒绝 → QuantLinear
(W4A8, 调 dynamic_quant) 的 quant 导出整体失败 (非 quant 导出因不调 dynamic_quant 而不受影响)。
注册本 converter (→ ge.DynamicQuant, torchair built-in) 后跳过 auto-callback, quant 导出通。
"""

import math
from typing import List, Optional

from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.ge_converter.converter_utils import *  # noqa: F401,F403  # Tensor/TensorSpec/torch_dtype_value_to_ge_type 等
from torchair.ge._ge_graph import (
    Tensor,
    TensorSpec,
    torch_dtype_value_to_ge_type,
    torch_dtype_value_to_ge_proto_type,
)


def convert_dynamic_quant(
    input_data: Tensor,
    smooth_scales: Optional[Tensor] = None,
    group_index: Optional[Tensor] = None,
    dst_type: Optional[int] = None,
    quant_mode: str = "pertoken",
    dst_type_max: float = 0.0,
    meta_outputs: List[TensorSpec] = None,
):
    # cann_lm::dynamic_quant dst_type 默认 None (CPU impl 忽略) -> int8 (acl code 1)
    if dst_type is None:
        dst_type = 1
    acl_dst_type = torch_dtype_value_to_ge_type(dst_type)
    if quant_mode == "pertoken" and math.isclose(dst_type_max, 0.0, rel_tol=1e-9):
        y, scale = ge.DynamicQuant(
            input_data,
            smooth_scales=smooth_scales,
            group_index=group_index,
            dst_type=acl_dst_type,
        )
    else:
        y, scale, offset = ge.DynamicQuantV2(
            input_data,
            smooth_scales,
            group_index,
            dst_type=acl_dst_type,
            is_symmetrical=True,
            quant_mode=quant_mode,
            dst_type_max=dst_type_max,
        )
    y.desc.dtype = torch_dtype_value_to_ge_proto_type(dst_type)
    return y, scale
