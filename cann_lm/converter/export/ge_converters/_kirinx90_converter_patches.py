"""KirinX90 GE converter patches.

环境变量 KIRINX90_CONVERTER_PATCHES=1 时由 ge_converter.py 加载。
覆盖 4 类 kirinx90 不支持的 GE 算子，用等价算子替换：

1. ge.MatMul      -> custom_op("MatMulV3")      (kirinx90 aicore 只有 V3)
2. ge.RotaryMul   -> ge.RotaryPositionEmbedding  (kirinx90 有此算子，走 arch35 分支)
3. ge.Muls        -> ge.Mul (标量转 Const tensor) (kirinx90 无 Muls engine)
4. ge.TensorMove  -> 跳过 (scatter 直发 ScatterNdUpdate)   (kirinx90 无 TensorMove engine)

注册机制: register_fx_node_ge_converter last-write-wins，
在 register_ge_converters() 之后调用 register_kirinx90_patches() 覆盖。
"""

import os
from typing import Any, Optional, Union

import torch


def register_kirinx90_patches() -> None:
    from torchair import register_fx_node_ge_converter
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge import attr
    from torchair._ge_concrete_graph.compat_ir import ge_op
    from torchair._ge_concrete_graph.ge_converter.converter_utils import (
        Tensor, TensorSpec, DataType, dtype_promote,
    )
    from torchair.ge._ge_graph import dont_prune_me

    def _matmul_v3(x1, x2):
        return ge_op(
            op_type="MatMulV3",
            inputs={"x1": x1, "x2": x2},
            outputs=["y"],
            attrs={
                "transpose_x1": attr.Bool(False),
                "transpose_x2": attr.Bool(False),
            },
        )

    def _batch_matmul_v3(x1, x2):
        return ge_op(
            op_type="BatchMatMulV3",
            inputs={"x1": x1, "x2": x2},
            outputs=["y"],
            attrs={
                "adj_x1": attr.Bool(False),
                "adj_x2": attr.Bool(False),
            },
        )

    # ── 1. MatMul -> MatMulV3 ──────────────────────────────────────────
    @register_fx_node_ge_converter(torch.ops.aten.mm.default)
    def _kirinx90_aten_mm(self: Tensor, mat2: Tensor, meta_outputs: TensorSpec = None):
        if self.dtype == DataType.DT_INT8 or mat2.dtype == DataType.DT_INT8:
            raise RuntimeError("aten.mm default ge_converter is not support int8 dtype!")
        self, mat2 = dtype_promote(self, mat2, target_dtype=meta_outputs.dtype)
        return _matmul_v3(self, mat2)

    @register_fx_node_ge_converter(torch.ops.aten.matmul.default)
    def _kirinx90_aten_matmul(self: Tensor, other: Tensor, meta_outputs: TensorSpec = None):
        if self.dtype == DataType.DT_INT8 or other.dtype == DataType.DT_INT8:
            raise RuntimeError("aten.matmul default ge_converter is not support int8 dtype!")
        if self.rank >= 3 or other.rank >= 3:
            self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
            return _batch_matmul_v3(self, other)
        self, other = dtype_promote(self, other, target_dtype=meta_outputs.dtype)
        return _matmul_v3(self, other)

    @register_fx_node_ge_converter(torch.ops.aten.bmm.default)
    def _kirinx90_aten_bmm(self: Tensor, mat2: Tensor, meta_outputs: TensorSpec = None):
        if self.dtype == DataType.DT_INT8 or mat2.dtype == DataType.DT_INT8:
            raise RuntimeError("aten.bmm default ge_converter is not support int8 dtype!")
        self, mat2 = dtype_promote(self, mat2, target_dtype=meta_outputs.dtype)
        return _batch_matmul_v3(self, mat2)

    # ── 2. RotaryMul -> RotaryPositionEmbedding ────────────────────────
    @register_fx_node_ge_converter(torch.ops.npu.npu_rotary_mul.default)
    def _kirinx90_rotary_mul(
        self: Tensor,
        r1: Tensor,
        r2: Tensor,
        rotary_mode: str = "half",
        rotate: Optional[Tensor] = None,
        meta_outputs: TensorSpec = None,
    ):
        modes = {"half": 0, "interleave": 1, "quarter": 2, "interleave-half": 3}
        if rotary_mode not in modes:
            raise NotImplementedError(
                f"rotary_mode={rotary_mode!r} not supported; only {list(modes)}"
            )
        mode = modes[rotary_mode]
        return ge.RotaryPositionEmbedding(self, r1, r2, mode=mode)

    # ── 3. Muls -> Mul (标量转 Const tensor) ───────────────────────────
    @register_fx_node_ge_converter(torch.ops.aten.mul.Tensor)
    def _kirinx90_aten_mul(
        self: Tensor,
        other: Union[Tensor, Any],
        meta_outputs: TensorSpec = None,
    ):
        if (
            not isinstance(other, Tensor)
            and isinstance(self, Tensor)
            and self.dtype in [DataType.DT_FLOAT16, DataType.DT_BF16]
        ):
            other_tensor = ge.Const(other, dtype=self.dtype)
            output = ge.Mul(self, other_tensor)
            return dtype_promote(output, target_dtype=meta_outputs.dtype)
        compute_dtype = meta_outputs.dtype
        self, other = dtype_promote(self, other, target_dtype=compute_dtype)
        output = ge.Mul(self, other)
        return dtype_promote(output, target_dtype=meta_outputs.dtype)

    # ── 4. TensorMove -> 跳过 (scatter 直发 ScatterNdUpdate) ──────────
    @register_fx_node_ge_converter(torch.ops.npu.npu_scatter_nd_update.default)
    def _kirinx90_scatter_nd_update(
        self: Tensor,
        indices: Tensor,
        updates: Tensor,
        meta_outputs: Any = None,
    ):
        del meta_outputs
        op = ge.ScatterNdUpdate(self, indices, updates)
        dont_prune_me(op)
        return op

    # ── 5. aten.slice.Tensor -> StridedSlice (non-V2, int32) ─────────
    #    kirinx90 的 ConstToAttrStridedSliceV2Fusion pass 是 forbidden closed
    #    (不可通过 fusion switch 关闭), 会把 StridedSliceV2 常量输入转为属性 →
    #    StridedSliceD (无 aicore kernel) → 落 host CPU → unknown shape → mobile 失败。
    #    改用 ge.StridedSlice (non-V2): ConstToAttr pass 只匹配 V2, 不影响 non-V2;
    #    non-V2 有 aicore kernel (strided_slice_apt, int32)。
    #    non-V2 无 axes 参数, 用 begin_mask/end_mask 跳过非切片维度。
    @register_fx_node_ge_converter(torch.ops.aten.slice.Tensor)
    def _kirinx90_aten_slice(
        self: Tensor,
        dim: int = 0,
        start: Optional[Union[int, Tensor]] = None,
        end: Optional[Union[int, Tensor]] = None,
        step: Union[int, Tensor] = 1,
        meta_outputs: TensorSpec = None,
    ):
        if start is None:
            start = 0
        if end is None:
            end = 2147483647
        if not isinstance(end, Tensor):
            if end == 9223372036854775807:
                end = 2147483647
            elif end > 2147483647:
                raise RuntimeError("StridedSlice does not support shapes exceeding INT32_MAX!")
        rank = self.rank
        mask = (1 << rank) - 1 - (1 << dim)
        begin_arr = [0] * rank
        end_arr = [0] * rank
        strides_arr = [1] * rank
        begin_arr[dim] = int(start) if not isinstance(start, Tensor) else start
        end_arr[dim] = int(end) if not isinstance(end, Tensor) else end
        strides_arr[dim] = int(step) if not isinstance(step, Tensor) else step
        begin_t = ge.Const(begin_arr, dtype=DataType.DT_INT32)
        end_t = ge.Const(end_arr, dtype=DataType.DT_INT32)
        strides_t = ge.Const(strides_arr, dtype=DataType.DT_INT32)
        result = ge.StridedSlice(
            self, begin_t, end_t, strides_t,
            begin_mask=mask, end_mask=mask,
        )
        # 预填 output shape (避免 IsUnknownShapeNode shape_range 空 → AICPU)
        if meta_outputs and hasattr(meta_outputs, 'size') and meta_outputs.size:
            del result.desc.shape.dim[:]
            result.desc.shape.dim.extend([int(d) for d in meta_outputs.size])
            result.desc.dtype = self.dtype
        return result

    # ── 5b. aten.transpose.int: 预填 output shape (避免 perm-in-input → AICPU) ──
    #    Transpose perm 在 input(Const) 时, GE InferShape 推出 shape 但不填 shape_range
    #    → IsUnknownShapeNode 判定 unknown → AICPU. 预填 output shape 避免.
    @register_fx_node_ge_converter(torch.ops.aten.transpose.int)
    def _kirinx90_transpose(self: Tensor, dim0: int = 0, dim1: int = 1, *, meta_outputs: TensorSpec = None):
        self = dtype_promote(self, target_dtype=meta_outputs.dtype)
        in_dims = list(self.desc.shape.dim) if hasattr(self, 'desc') and self.desc.shape.dim else []
        rank = len(in_dims) if in_dims else (len(meta_outputs.size) if meta_outputs and hasattr(meta_outputs, 'size') else 2)
        perm = list(range(rank))
        d0 = dim0 if dim0 >= 0 else dim0 + rank
        d1 = dim1 if dim1 >= 0 else dim1 + rank
        perm[d0], perm[d1] = perm[d1], perm[d0]
        perm_t = ge.Const(perm, dtype=DataType.DT_INT64)
        result = ge.Transpose(self, perm_t)
        if meta_outputs and hasattr(meta_outputs, 'size') and meta_outputs.size:
            del result.desc.shape.dim[:]
            result.desc.shape.dim.extend([int(d) for d in meta_outputs.size])
            result.desc.dtype = self.dtype
        return result

    # ── 5c. (移除 arange patch — Range 输出 DT_FLOAT 致 GatherV2 indices 不支持) ──

    # ── 5d. (移除 aten.to.dtype Cast patch — 改变了 indices dtype 致 GatherV2 找不到 engine) ──

    # ── 6. aten.cat -> ConcatD (非 V2) ────────────────────────────────
    #    kirinx90 aicore 只有 ConcatD (非 ConcatV2); V2 全落 aicpu。
    @register_fx_node_ge_converter(torch.ops.aten.cat.default)
    def _kirinx90_aten_cat(tensors: list, dim: int = 0, meta_outputs: TensorSpec = None):
        tensors = [dtype_promote(arg, target_dtype=meta_outputs.dtype) for arg in tensors]
        return ge.ConcatD(tensors, concat_dim=dim, N=len(tensors))

    # ── 7. Conv2D -> Conv2DV2 (kirinx90 aicore 只有 Conv2DV2, 无 plain Conv2D) ──
    @register_fx_node_ge_converter(torch.ops.aten.conv1d.default)
    def _kirinx90_conv1d(
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Any = (1,),
        padding: Any = (0,),
        dilation: Any = (1,),
        groups: int = 1,
        *,
        meta_outputs: TensorSpec = None,
    ):
        s = stride[0] if isinstance(stride, (list, tuple)) and stride else 1
        p = padding[0] if isinstance(padding, (list, tuple)) and padding else 0
        d = dilation[0] if isinstance(dilation, (list, tuple)) and dilation else 1
        x4 = ge.Unsqueeze(x, axes=[2])
        w4 = ge.Unsqueeze(weight, axes=[2])
        # Conv2DV2 InferShape 要求 NCHW format (不是 ND), 否则 group check 用错索引
        x4.desc.layout = "NCHW"
        w4.desc.layout = "NCHW"
        w_dims = list(weight.desc.shape.dim) if hasattr(weight, "desc") and weight.desc.shape.dim else [1]
        out_channels = w_dims[0] if w_dims else 1
        bias_in = bias if bias is not None else ge.Const([0] * out_channels, dtype=DataType.DT_FLOAT16)
        out4 = ge_op(
            op_type="Conv2DV2",
            inputs={"x": x4, "filter": w4},
            outputs=["y"],
            attrs={
                "strides": attr.ListInt([1, 1, 1, s]),
                "pads": attr.ListInt([0, 0, p, p]),
                "dilations": attr.ListInt([1, 1, 1, d]),
                "groups": attr.Int(groups),
                "data_format": attr.Str("NCHW"),
                "offset_x": attr.Int(0),
            },
        )
        out4.desc.layout = "NCHW"
        return ge.Squeeze(out4, axis=[2])

    # ── 8. FIA v2: sparse_mode=0 + mask [*,2048,2048] → sparse_mode=4 (BAND) ──
    #    kirinx90 arch31 check: sm=0+mask 要求 mask shape=(B,S1,S2), 但 qwen3.5
    #    sliding attention 的 mask 是 [1,1,2048,2048] (固定 kv_max 方阵), S1=128≠2048 → 失败.
    #    sm=4 (BAND) 要求 mask=(2048,2048) 固定大小 → 匹配. 只影响 kirinx90, 不影响 910.
    #    ⚠️ gemma4 的 mask 是 [1,1,128,2048] (非方阵 S1×S2), sm=0 匹配 (B,S1,S2) → 不改.
    #    converter 时无法读 mask shape, 用 env KIRINX90_FIA_SPARSE_MODE_4=1 显式开启 (qwen3.5 用).
    from ._fused_infer_attention_score_v2_converter import convert_fused_infer_attention_score_v2

    @register_fx_node_ge_converter(torch.ops.npu.npu_fused_infer_attention_score_v2.default)
    def _kirinx90_fia_v2(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        atten_mask: Any = None,
        actual_seq_qlen: Any = None,
        actual_seq_kvlen: Any = None,
        block_table: Any = None,
        num_query_heads: int = 1,
        num_key_value_heads: int = 0,
        softmax_scale: float = 1.0,
        pre_tokens: int = 2147483647,
        next_tokens: int = 2147483647,
        input_layout: str = "BSH",
        sparse_mode: int = 0,
        block_size: int = 0,
        inner_precise: int = 0,
        return_softmax_lse: bool = False,
        meta_outputs: Any = None,
    ):
        # 仅 env KIRINX90_FIA_SPARSE_MODE_4=1 时改 sm=0→4 (qwen3.5 方阵 mask 用)
        # gemma4 非方阵 mask [1,1,128,2048] 用 sm=0, 不改
        if sparse_mode == 0 and atten_mask is not None:
            if os.environ.get("KIRINX90_FIA_SPARSE_MODE_4", "") == "1":
                sparse_mode = 4
                if pre_tokens > 2048:
                    pre_tokens = 2048
        return convert_fused_infer_attention_score_v2(
            query, key, value,
            atten_mask=atten_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            block_table=block_table,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            softmax_scale=softmax_scale,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            input_layout=input_layout,
            sparse_mode=sparse_mode,
            block_size=block_size,
            inner_precise=inner_precise,
            return_softmax_lse=return_softmax_lse,
            meta_outputs=meta_outputs,
        )
