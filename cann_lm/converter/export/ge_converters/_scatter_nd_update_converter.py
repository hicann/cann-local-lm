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

"""cann_lm::scatter_nd_update(_) GE converter -> ge.ScatterNdUpdate.

Maps directly to ge.ScatterNdUpdate (standard multi-dim coordinate scatter), bypassing
the aten.index_copy InplaceUpdate+Transpose path. out-of-place adds ge.TensorMove (in-place
precision); in-place adds dont_prune_me (prevent graph pruning). Overrides torchair's
custom/npu_scatter_nd_update.py so behavior is controlled in the torch_npu-absent converter venv.
Kirinx90 走 KIRIN_CONVERTER_PATCHES 机制 (见 _kirin_converter_patches.py 第 4 条: 直发
ScatterNdUpdate 跳过 TensorMove; kirin 无 TensorMove kernel, 见 kirin_compile_workarounds.md B2).
"""
from typing import Any


def convert_scatter_nd_update(
    self: Any,
    indices: Any,
    updates: Any,
    meta_outputs: Any = None,
) -> Any:
    """npu_scatter_nd_update (out-of-place) -> ge.TensorMove + ge.ScatterNdUpdate."""
    del meta_outputs
    from torchair._ge_concrete_graph import ge_apis as ge
    copy = ge.TensorMove(self)
    return ge.ScatterNdUpdate(copy, indices, updates)


def convert_scatter_nd_update_(
    self: Any,
    indices: Any,
    updates: Any,
    meta_outputs: Any = None,
) -> Any:
    """npu_scatter_nd_update_ (in-place) -> ge.ScatterNdUpdate + dont_prune_me."""
    del meta_outputs
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import dont_prune_me
    op = ge.ScatterNdUpdate(self, indices, updates)
    dont_prune_me(op)
    return op
