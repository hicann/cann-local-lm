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

"""npu::npu_scatter_nd_update(_) GE converter -> ge.ScatterNdUpdate.

cann_lm 自带 converter, 直接映射到 ge.ScatterNdUpdate (标准多维坐标 scatter 语义),
不绕 aten.index_copy 的 InplaceUpdate+Transpose 路径, 使 IR 阶段即出现 ScatterNdUpdate
算子. out-of-place 版加 ge.TensorMove (避免 in-place 精度问题), in-place 版加
dont_prune_me (防图剪枝).

**语义约定**: ge.ScatterNdUpdate 按标准多维坐标 scatter (indices 每行是一个完整坐标,
定位 cache 前若干维). 调用方 (如 gemma4) 需构造 [num, ndim] 多维 indices, 而非 1D 单维
索引. 这与 cann_lm CPU impl 的 indices.dim()==1 else 分支 (逐坐标点写) 一致,
CPU eager 与 GE 导出语义对齐.

覆盖 torchair custom/npu_scatter_nd_update.py: 逻辑一致 (均产出 ge.ScatterNdUpdate),
cann_lm 显式持有此映射 (与 _quant_matmul_converter/_chunk_gated_delta_rule_converter
一致的覆盖模式), 确保 converter venv (torch_npu absent) 下行为可控、不依赖 torchair
注册顺序.

注册由 ge_converter.py:register_ge_converters 调
register_fx_node_ge_converter(torch.ops.npu.npu_scatter_nd_update.default)(convert_scatter_nd_update)
register_fx_node_ge_converter(torch.ops.npu.npu_scatter_nd_update_.default)(convert_scatter_nd_update_)
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
