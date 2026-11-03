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

"""cann_local_lm::scatter_nd_update - KV cache scatter-write (schema + meta + CPU eager kernel).

注册到 npu 命名空间 (torch.ops.cann_local_lm.scatter_nd_update), 替代 aten.index_copy / aten.index_put
的 InplaceUpdate + Transpose 方案. GE 转换器 -> ge.ScatterNdUpdate (in-place 版 1 个算子, 无 Transpose; out-of-place 版多 1 个 ge.TensorMove).
与图灵参考实现 (cann-recipes-infer) 对齐: torch_npu.npu_scatter_nd_update_(cache, slot_mapping, updates).

已落地 gemma4: modeling_gemma4.py 的 scatter_kv_to_cache helper 构造 [b*n*seq,3] 多维
indices (b,n,kv_pos) + [b*n*seq,d] updates, 调本算子写入 BNSD cache, 等价 index_copy(2,...).
GE converter 由 cann_local_lm 自带 (converter/export/_scatter_nd_update_converter.py, 覆盖 torchair),
直接产出 ge.ScatterNdUpdate (标准多维 scatter 语义, 与 CPU else 分支一致).

用法: scatter KV cache (按 slot_mapping 把 updates 写入 cache 的指定位置).
  cache:     [b, n, kv_max, d]  (pre-allocated buffer)
  indices:   [num_slots, ndim]   (写入位置, 每行一个坐标)
  updates:   [b, n, seq, d]     (新 KV, 按 indices 写入 cache)

注意: torchair custom/ 有 npu_scatter_nd_update 的 converter (注册到 npu 命名空间),
但本算子用 npu 命名空间 (stubs 环境自己注册 schema+fake+CPU eager).
"""

import torch

from ._library import NPU_LIB, safe_define, _HAS_TORCH_NPU, register_cpu_impl

_SCATTER_SCHEMA = (
    "scatter_nd_update(Tensor self, Tensor indices, Tensor updates) -> Tensor out"
)

# in-place 版本 (torchair _optimize_pass 的 re-inplace 会尝试访问, 必须注册)
_SCATTER_INPLACE_SCHEMA = (
    "scatter_nd_update_(Tensor(a!) self, Tensor indices, Tensor updates) -> Tensor(a!)"
)

if NPU_LIB is not None:
    safe_define(_SCATTER_SCHEMA)
    safe_define(_SCATTER_INPLACE_SCHEMA)

if not _HAS_TORCH_NPU:

    @torch.library.register_fake("cann_local_lm::scatter_nd_update")
    def _scatter_meta(self, indices, updates):
        return torch.empty_like(self)

    @torch.library.register_fake("cann_local_lm::scatter_nd_update_")
    def _scatter_inplace_meta(self, indices, updates):
        return torch.empty_like(self)

    @register_cpu_impl("cann_local_lm::scatter_nd_update")
    def _scatter_cpu(self, indices, updates):
        """CPU eager: 用 index_copy 实现 scatter_nd_update (按 indices 坐标写入)."""
        out = self.clone()
        if indices.dim() == 1:
            out.index_copy_(2, indices.to(torch.int64), updates)
        else:
            for i in range(indices.shape[0]):
                idx = indices[i]
                out[tuple(idx.tolist())] = updates[i]
        return out

    @register_cpu_impl("cann_local_lm::scatter_nd_update_")
    def _scatter_inplace_cpu(self, indices, updates):
        """CPU eager (in-place): 直接修改 self."""
        if indices.dim() == 1:
            self.index_copy_(2, indices.to(torch.int64), updates)
        else:
            for i in range(indices.shape[0]):
                idx = indices[i]
                self[tuple(idx.tolist())] = updates[i]
        return self
