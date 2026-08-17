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

"""aten.index_put GE converter (从 ge_converter.py 抽出).

qwen3.5 的 ``past_key_values.update`` 用 index_put (不是 index_copy). indices 是 list of
(tensor | None), 找非 None 的 dim + index, 复用 ``InplaceUpdate + Transpose`` (与 index_copy 同).

**必须自定义 converter**: aten.index_put 无 torch decomposition, torchair 自带 converter 仅占位
``NotImplementedError``. 不支持 accumulate=True / all-None indices (raise NotImplementedError).

注册由 ``ge_converter.py:register_ge_converters`` 调
``register_fx_node_ge_converter(torch.ops.aten.index_put.default)(convert_index_put)``.
"""
from typing import Any


def convert_index_put(
    self: Any,
    indices: Any,
    values: Any,
    *,
    accumulate: bool = False,
    meta_outputs: Any = None,
) -> Any:
    """aten.index_put -> InplaceUpdate + Transpose (找非 None dim, 同 index_copy)."""
    del meta_outputs
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import DataType
    from torchair._ge_concrete_graph.ge_converter.converter_utils import dtype_promote
    if accumulate:
        raise NotImplementedError("index_put with accumulate=True not supported")
    dim = None
    index = None
    for i, idx in enumerate(indices):
        if idx is not None:
            dim = i
            index = idx
            break
    if dim is None:
        raise NotImplementedError("index_put with all-None indices not supported")
    if index.dtype != DataType.DT_INT32:
        index = ge.Cast(index, dst_type=int(DataType.DT_INT32))
    ndim = self.rank
    if dim < 0:
        dim += ndim
    perm = [dim] + [i for i in range(ndim) if i != dim]
    inverse_perm = [0] * ndim
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    perm = dtype_promote(perm, target_dtype=DataType.DT_INT64)
    inverse_perm = dtype_promote(inverse_perm, target_dtype=DataType.DT_INT64)
    moved = ge.TensorMove(self)
    transposed_self = ge.Transpose(moved, perm)
    transposed_values = ge.Transpose(values, perm)
    transposed_values = ge.Cast(transposed_values, dst_type=int(self.dtype))
    updated = ge.InplaceUpdate(transposed_self, index, transposed_values)
    return ge.Transpose(updated, inverse_perm)
