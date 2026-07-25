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

"""TorchAir converters for the three demo custom ops."""

from __future__ import annotations

from typing import Any

import torch

from cann_lm.models.fused_ops import ops as _backend_ops  # Registers cann_lm ops.


def register_ge_converters() -> None:
    from torchair import register_fx_node_ge_converter
    from torchair.ge import custom_op
    from torchair._ge_concrete_graph import ge_apis as ge

    from ._quant_matmul_converter import convert_quant_matmul
    register_fx_node_ge_converter(torch.ops.cann_lm.quant_matmul.default)(convert_quant_matmul)

    @register_fx_node_ge_converter(torch.ops.aten.softplus.default)
    def _convert_softplus(x: Any, meta_outputs: Any = None) -> Any:
        del meta_outputs
        return ge.Softplus(x)

    @register_fx_node_ge_converter(torch.ops.aten.eye.default)
    def _convert_eye(n: Any, m: Any = None, *, dtype: Any = None, device: Any = None,
                     pin_memory: bool = False, meta_outputs: Any = None) -> Any:
        del device, pin_memory, meta_outputs
        import torch as _torch
        rows = int(n)
        cols = int(m) if m is not None else rows
        dt = _torch.float32 if dtype is None else dtype
        eye_tensor = _torch.eye(rows, cols, dtype=dt)
        return ge.Const(eye_tensor)

    @register_fx_node_ge_converter(torch.ops.cann_lm.rope_in_place.default)
    def _convert_rope_in_place(
        input: Any,
        sin_cache: Any,
        cos_cache: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return custom_op(
            "RopeInPlace",
            inputs={"input": input, "sin_cache": sin_cache, "cos_cache": cos_cache},
            outputs=["output"],
        )

    @register_fx_node_ge_converter(torch.ops.cann_lm.beam_search.default)
    def _convert_beam_search(
        log_probs: Any,
        top_tokens: Any,
        top_probs: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return custom_op(
            "BeamSearch",
            inputs={
                "log_probs": log_probs,
                "top_tokens": top_tokens,
                "top_probs": top_probs,
            },
            outputs=["out_token_ids", "out_token_index", "out_log_probs"],
        )

    @register_fx_node_ge_converter(torch.ops.cann_lm.fused_infer_attention_score.default)
    def _convert_fused_infer_attention_score(
        query: Any,
        key: Any,
        value: Any,
        *,
        pse_shift: Any = None,
        atten_mask: Any = None,
        actual_seq_lengths: Any = None,
        actual_seq_lengths_kv: Any = None,
        dequant_scale1: Any = None,
        quant_scale1: Any = None,
        dequant_scale2: Any = None,
        quant_scale2: Any = None,
        quant_offset2: Any = None,
        antiquant_scale: Any = None,
        antiquant_offset: Any = None,
        block_table: Any = None,
        query_padding_size: Any = None,
        kv_padding_size: Any = None,
        key_antiquant_scale: Any = None,
        key_antiquant_offset: Any = None,
        value_antiquant_scale: Any = None,
        value_antiquant_offset: Any = None,
        key_shared_prefix: Any = None,
        value_shared_prefix: Any = None,
        actual_shared_prefix_len: Any = None,
        query_rope: Any = None,
        key_rope: Any = None,
        key_rope_antiquant_scale: Any = None,
        num_heads: int = 1,
        scale: float = 1.0,
        pre_tokens: int = 2147483647,
        next_tokens: int = 2147483647,
        input_layout: str = "BSH",
        num_key_value_heads: int = 0,
        sparse_mode: int = 0,
        inner_precise: int = 0,
        block_size: int = 0,
        antiquant_mode: int = 0,
        softmax_lse_flag: bool = False,
        key_antiquant_mode: int = 0,
        value_antiquant_mode: int = 0,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        key_list = [key]
        value_list = [value]
        return ge.FusedInferAttentionScore(
            query, key_list, value_list,
            pse_shift=pse_shift,
            atten_mask=atten_mask,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            dequant_scale1=dequant_scale1,
            quant_scale1=quant_scale1,
            dequant_scale2=dequant_scale2,
            quant_scale2=quant_scale2,
            quant_offset2=quant_offset2,
            antiquant_scale=antiquant_scale,
            antiquant_offset=antiquant_offset,
            block_table=block_table,
            query_padding_size=query_padding_size,
            kv_padding_size=kv_padding_size,
            key_antiquant_scale=key_antiquant_scale,
            key_antiquant_offset=key_antiquant_offset,
            value_antiquant_scale=value_antiquant_scale,
            value_antiquant_offset=value_antiquant_offset,
            key_shared_prefix=key_shared_prefix,
            value_shared_prefix=value_shared_prefix,
            actual_shared_prefix_len=actual_shared_prefix_len,
            query_rope=query_rope,
            key_rope=key_rope,
            key_rope_antiquant_scale=key_rope_antiquant_scale,
            num_heads=num_heads,
            scale=scale,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            input_layout=input_layout,
            num_key_value_heads=num_key_value_heads,
            sparse_mode=sparse_mode,
            inner_precise=inner_precise,
            block_size=block_size,
            antiquant_mode=antiquant_mode,
            softmax_lse_flag=softmax_lse_flag,
            key_antiquant_mode=key_antiquant_mode,
            value_antiquant_mode=value_antiquant_mode,
            dequant_scale_query=None,
            learnable_sink=None,
            q_start_idx=None,
            kv_start_idx=None,
            query_quant_mode=0,
            pse_type=0,
            out_dtype=0,
        )

    @register_fx_node_ge_converter(torch.ops.cann_lm.rms_norm.default)
    def _convert_rms_norm(
        x: Any,
        gamma: Any,
        *,
        epsilon: float = 1e-6,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        return ge.RmsNorm(x, gamma, epsilon=epsilon)

    @register_fx_node_ge_converter(torch.ops.cann_lm.add_rms_norm.default)
    def _convert_add_rms_norm(
        x1: Any,
        x2: Any,
        gamma: Any,
        *,
        epsilon: float = 1e-6,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        # ge.AddRmsNorm(x1, x2, gamma, *, epsilon) -> (y, rstd, x=x1+x2)
        return ge.AddRmsNorm(x1, x2, gamma, epsilon=epsilon)

    @register_fx_node_ge_converter(torch.ops.cann_lm.rotary_mul.default)
    def _convert_rotary_mul(
        x: Any,
        cos: Any,
        sin: Any,
        *,
        rotary_mode: str = "half",
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        if rotary_mode != "half":
            raise NotImplementedError(
                f"rotary_mode={rotary_mode!r} not mapped; "
                "CANN RotaryMul only supports half-rotation"
            )
        # ge.RotaryMul(x, r1=cos, r2=sin) -> y
        return ge.RotaryMul(x, cos, sin)

    @register_fx_node_ge_converter(torch.ops.aten.type_as.default)
    def _convert_type_as(x: Any, dtype_or_tensor: Any, meta_outputs: Any = None) -> Any:
        del meta_outputs
        import torch as _torch
        if isinstance(dtype_or_tensor, _torch.Tensor):
            dt = dtype_or_tensor.dtype
        else:
            dt = dtype_or_tensor
        _GE_DTYPE_MAP = {
            _torch.float16: 1, _torch.float32: 0, _torch.bfloat16: 27,
            _torch.int64: 9, _torch.int32: 3, _torch.bool: 12,
        }
        return ge.Cast(x, dst_type=_GE_DTYPE_MAP.get(dt, 0))

    @register_fx_node_ge_converter(torch.ops.aten.index_copy.default)
    def _convert_index_copy(
        self: Any,
        dim: int,
        index: Any,
        source: Any,
        meta_outputs: Any = None,
    ) -> Any:
        """aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor

        Implemented via InplaceUpdate + transpose: ``InplaceUpdate`` only supports
        dim=0, so transpose self/source to move ``dim`` to axis 0, InplaceUpdate
        (replace indices), then transpose back. Supports arbitrary 1D index
        (not limited to contiguous arange).
        """
        del meta_outputs
        from torchair.ge._ge_graph import DataType
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
        # TensorMove to avoid in-place precision issue (same pattern as index_add.py)
        moved = ge.TensorMove(self)
        transposed_self = ge.TransposeD(moved, perm=perm)
        transposed_source = ge.TransposeD(source, perm=perm)
        # Force dtype alignment: InplaceUpdate requires x and v to have the same dtype.
        # Use self.dtype (known) rather than transposed_self.dtype (UNDEFINED before infershape).
        transposed_source = ge.Cast(transposed_source, dst_type=int(self.dtype))
        updated = ge.InplaceUpdate(transposed_self, index, transposed_source)
        return ge.TransposeD(updated, perm=inverse_perm)
