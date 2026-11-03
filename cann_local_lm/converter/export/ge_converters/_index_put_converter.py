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

"""aten.index_put GE converter — find the non-None dim, reuse InplaceUpdate + Transpose (same as index_copy).

torchair's converter is a placeholder (NotImplementedError). Does not support accumulate=True / all-None indices.
Registered by ge_converter.register_ge_converters.
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
    """aten.index_put -> InplaceUpdate + Transpose (find the non-None dim, same as index_copy)."""
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
