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

"""aten.conv1d GE converter -> Unsqueeze + Conv2D + Squeeze.

Custom converter required: decomposition leaves the Conv2D inputDesc empty (torchair
auto_generated only sets name="y"), and GE infershape won't infer empty desc -> OM compile
fails. This manually fills Unsqueeze/Conv2D outputDesc shape/dtype. Registered by ge_converter.register_ge_converters.
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
    """aten.conv1d -> Unsqueeze + Conv2D + Squeeze; manually fill outputDesc to avoid GE infershape on empty desc."""
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.ge_converter.aten.convolution import _npu_convolution
    s = stride[0] if isinstance(stride, (list, tuple)) and stride else 1
    p = padding[0] if isinstance(padding, (list, tuple)) and padding else 0
    d = dilation[0] if isinstance(dilation, (list, tuple)) and dilation else 1
    x4 = ge.Unsqueeze(x, axes=[2])       # [B,C,L] -> [B,C,1,L]
    w4 = ge.Unsqueeze(weight, axes=[2])  # [O,C,K] -> [O,C,1,K]
    # Manually fill Unsqueeze outputDesc shape/dtype so the downstream Conv2D inputDesc (CopyFrom)
    # is non-empty, avoiding GE infershape skipping empty desc (torchair auto_generated only sets name="y")
    w_dims = list(weight.desc.shape.dim)  # [O,C,K] (constant has shape.dim)
    # x.desc.shape.dim is empty (boundary only has _meta); infer L_in from meta_outputs.size [B,C,L_out]
    mo = meta_outputs[0] if isinstance(meta_outputs, (list, tuple)) else meta_outputs
    mo_size = list(getattr(mo, "size", []) or [])
    x_dims = []
    if len(mo_size) == 3 and len(w_dims) == 3:
        B, C, L_out = mo_size[0], mo_size[1], mo_size[2]
        K = w_dims[2]
        L_in = (L_out - 1) * s + K - 2 * p
        x_dims = [B, C, L_in]
    # Fill x4 (Unsqueeze output) shape [B,C,1,L_in] + dtype (layout set by _npu_conv2d's specific_op_input_layout)
    if len(x_dims) == 3:
        del x4.desc.shape.dim[:]
        x4.desc.shape.dim.extend([x_dims[0], x_dims[1], 1, x_dims[2]])
        x4.desc.dtype = x.desc.dtype
        x4.desc.layout = "NCHW"
    # Fill w4 (Unsqueeze output) shape [O,C,1,K] + dtype
    if len(w_dims) == 3:
        del w4.desc.shape.dim[:]
        w4.desc.shape.dim.extend([w_dims[0], w_dims[1], 1, w_dims[2]])
        w4.desc.dtype = weight.desc.dtype
        w4.desc.layout = "NCHW"
    out4 = _npu_convolution(x4, w4, bias, [1, s], [0, p], [1, d], groups, meta_outputs)
    # Fill Conv2D outputDesc shape [B,O,1,L_out] + dtype
    if len(mo_size) == 3 and len(w_dims) == 3:
        del out4.desc.shape.dim[:]
        out4.desc.shape.dim.extend([mo_size[0], w_dims[0], 1, mo_size[2]])
        out4.desc.dtype = x.desc.dtype
        out4.desc.layout = "NCHW"
    return ge.Squeeze(out4, axis=[2])
