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

"""aten.conv1d GE converter (从 ge_converter.py 抽出).

Maps ``torch.ops.aten.conv1d`` -> ``Unsqueeze + _npu_convolution (Conv2D) + Squeeze``
(1D [B,C,L] -> 2D [B,C,1,L] -> Conv2D -> Squeeze 回 [B,C,L]).

**必须自定义 converter** (不能走 whitelist 分解):
- aten.conv1d 无 torch decomposition, whitelist 条目对它是空操作;
- 走分解 (aten.convolution + Unsqueeze/Squeeze) 会让下游 Conv2D inputDesc 空 (torchair auto_generated
  只设 name="y"), GE infershape 不推空 desc -> OM 编译 ``Prepare Graph infershape failed``.
- 本 converter 手动填 Unsqueeze/Conv2D 的 outputDesc shape/dtype (从 weight.desc.shape.dim + meta_outputs
  反推 L_in), 让下游 inputDesc (CopyFrom) 非空, 避开 infershape 空 desc 问题.

注册由 ``ge_converter.py:register_ge_converters`` 调
``register_fx_node_ge_converter(torch.ops.aten.conv1d.default)(convert_conv1d)``.
"""
from typing import Any


def convert_conv1d(
    x: Any,
    weight: Any,
    bias: Any = None,
    stride: Any = (1,),
    padding: Any = (0,),
    dilation: Any = (1,),
    groups: int = 1,
    *,
    meta_outputs: Any = None,
) -> Any:
    """aten.conv1d -> Unsqueeze + Conv2D + Squeeze, 手动填 outputDesc 避开 GE infershape 空 desc."""
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.ge_converter.aten.convolution import _npu_convolution
    s = stride[0] if isinstance(stride, (list, tuple)) and stride else 1
    p = padding[0] if isinstance(padding, (list, tuple)) and padding else 0
    d = dilation[0] if isinstance(dilation, (list, tuple)) and dilation else 1
    x4 = ge.Unsqueeze(x, axes=[2])       # [B,C,L] -> [B,C,1,L]
    w4 = ge.Unsqueeze(weight, axes=[2])  # [O,C,K] -> [O,C,1,K]
    # 手动填 Unsqueeze outputDesc 的 shape/dtype, 让下游 Conv2D inputDesc (CopyFrom) 非空,
    # 避开 GE infershape 不推空 desc (torchair auto_generated 只设 name="y")
    w_dims = list(weight.desc.shape.dim)  # [O,C,K] (constant 有 shape.dim)
    # x.desc.shape.dim 空 (boundary 只有 _meta), 从 meta_outputs.size [B,C,L_out] 反推 L_in
    mo = meta_outputs[0] if isinstance(meta_outputs, (list, tuple)) else meta_outputs
    mo_size = list(getattr(mo, "size", []) or [])
    x_dims = []
    if len(mo_size) == 3 and len(w_dims) == 3:
        B, C, L_out = mo_size[0], mo_size[1], mo_size[2]
        K = w_dims[2]
        L_in = (L_out - 1) * s + K - 2 * p
        x_dims = [B, C, L_in]
    # 填 x4 (Unsqueeze output) shape [B,C,1,L_in] + dtype (layout 由 _npu_conv2d 的 specific_op_input_layout 设)
    if len(x_dims) == 3:
        del x4.desc.shape.dim[:]
        x4.desc.shape.dim.extend([x_dims[0], x_dims[1], 1, x_dims[2]])
        x4.desc.dtype = x.desc.dtype
        x4.desc.layout = "NCHW"
    # 填 w4 (Unsqueeze output) shape [O,C,1,K] + dtype
    if len(w_dims) == 3:
        del w4.desc.shape.dim[:]
        w4.desc.shape.dim.extend([w_dims[0], w_dims[1], 1, w_dims[2]])
        w4.desc.dtype = weight.desc.dtype
        w4.desc.layout = "NCHW"
    out4 = _npu_convolution(x4, w4, bias, [1, s], [0, p], [1, d], groups, meta_outputs)
    # 填 Conv2D outputDesc shape [B,O,1,L_out] + dtype
    if len(mo_size) == 3 and len(w_dims) == 3:
        del out4.desc.shape.dim[:]
        out4.desc.shape.dim.extend([mo_size[0], w_dims[0], 1, mo_size[2]])
        out4.desc.dtype = x.desc.dtype
        out4.desc.layout = "NCHW"
    return ge.Squeeze(out4, axis=[2])
