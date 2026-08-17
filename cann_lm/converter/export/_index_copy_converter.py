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

"""aten.index_copy GE converter (从 ge_converter.py 抽出).

KV cache scatter-write (qwen3.5/gemma4/qwen3 的 ``cache.index_copy(2, new_kv_cache_pos, key_states)``).

**必须自定义 converter**: aten.index_copy 无 torch decomposition (whitelist 条目对它空操作),
torchair 自带 converter 仅占位 ``NotImplementedError``. 本 converter 用 ``InplaceUpdate + Transpose``
实现 scatter-write (InplaceUpdate 只支持 dim=0, 故 transpose 把 ``dim`` 移到 axis 0, 写完再转回).

注册由 ``ge_converter.py:register_ge_converters`` 调
``register_fx_node_ge_converter(torch.ops.aten.index_copy.default)(convert_index_copy)``.
"""
from typing import Any


def convert_index_copy(
    self: Any,
    dim: int,
    index: Any,
    source: Any,
    meta_outputs: Any = None,
) -> Any:
    """aten.index_copy -> InplaceUpdate + Transpose (dim!=0 时转置 dim 到 axis 0)."""
    del meta_outputs
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import DataType
    from torchair._ge_concrete_graph.ge_converter.converter_utils import dtype_promote
    # InplaceUpdate requires DT_INT32 indices (see REG_OP InplaceUpdate input)
    if index.dtype != DataType.DT_INT32:
        index = ge.Cast(index, dst_type=int(DataType.DT_INT32))
    ndim = self.rank
    if dim < 0:
        dim += ndim
    # perm moves `dim` to position 0: [dim, 0, 1, ..., dim-1, dim+1, ..., ndim-1]
    perm = [dim] + [i for i in range(ndim) if i != dim]
    inverse_perm = [0] * ndim
    for i, p in enumerate(perm):
        inverse_perm[p] = i
    # Use ge.Transpose (perm as Const input) instead of ge.TransposeD (perm as attr):
    # Ascend950 OPP ships Transpose but has no TransposeD engine, so TransposeD fails
    # engine selection on 950. Transpose(perm=Const) stays Transpose (950-supported);
    # 910 has both. dtype_promote -> DT_INT64 Const perm tensor (matches torchair default).
    perm = dtype_promote(perm, target_dtype=DataType.DT_INT64)
    inverse_perm = dtype_promote(inverse_perm, target_dtype=DataType.DT_INT64)
    # TensorMove to avoid in-place precision issue (same pattern as index_add.py)
    moved = ge.TensorMove(self)
    transposed_self = ge.Transpose(moved, perm)
    transposed_source = ge.Transpose(source, perm)
    # Force dtype alignment: InplaceUpdate requires x and v to have the same dtype.
    # Use self.dtype (known) rather than transposed_self.dtype (UNDEFINED before infershape).
    transposed_source = ge.Cast(transposed_source, dst_type=int(self.dtype))
    updated = ge.InplaceUpdate(transposed_self, index, transposed_source)
    return ge.Transpose(updated, inverse_perm)
