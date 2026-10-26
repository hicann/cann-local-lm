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

"""Weight int4 materialization (W4A8 s4, symmetric, per-group).

Converts a fp16/bf16/fp32 nn.Linear weight [out, in] into a nibble-packed int4
payload (stored as int8 [out, in//2], low nibble = even k index, two's-complement
4-bit signed [-8, 7]) + per-group fp32 scale [out, in//group] (K-axis), offset=0
(symmetric). Output layout matches the ``cann_lm::quant_matmul`` x2 input.
"""

import torch


_SCALE_MASK = 0x00000000FFFFE000  # bitmask: clear high 32 bits + truncate mantissa low 13 bits (storage fmt, see quant_design.md)


def transdata_scale(scale_fp32: torch.Tensor) -> torch.Tensor:
    """fp32 -> int64 bit-pattern & _SCALE_MASK (truncates mantissa low 13 bits, storage format).

    torch has no uint64; int64 holds the same bit pattern. Inverse: restore_scale.
    """
    bits = scale_fp32.contiguous().view(torch.int32).to(torch.int64)
    return bits & _SCALE_MASK


def restore_scale(scale_packed: torch.Tensor) -> torch.Tensor:
    """int64 (bit-pattern + mask) -> fp32 (mantissa low 13 bits zeroed, reduced precision)."""
    return scale_packed.to(torch.int32).view(torch.float32)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Nibble-pack signed int4 values (stored as int8) into [n, k//2] int8.

    Input: ``q`` int8 ``[..., k]`` holding signed int4 values in [-8, 7].
    Output: int8 ``[..., k//2]`` — byte at index i holds int4 for k=2i (low
    nibble) and k=2i+1 (high nibble), two's-complement 4-bit (``value & 0xF``).
    Matches the ``cann_lm::quant_matmul`` x2 input layout (nibble-packed int4).
    """
    if q.shape[-1] % 2 != 0:
        raise ValueError(f"last dim {q.shape[-1]} must be even for int4 nibble-pack")
    q = q.to(torch.int8)
    nib_even = q[..., 0::2] & 0x0F
    nib_odd = q[..., 1::2] & 0x0F
    return ((nib_odd << 4) | nib_even).to(torch.int8)


def pack_int4_int32(q: torch.Tensor) -> torch.Tensor:
    """CANN int4pack: 8 个 int4 沿最后一维交叠进 1 个 int32 (int32 ``[..., n//8]``).

    Input: ``q`` int8 ``[..., n]`` holding signed int4 values in [-8, 7].
    Output: int32 ``[..., n//8]`` — int32 at index i holds int4 for n=8i (bit 0-3),
    n=8i+1 (bit 4-7), ..., n=8i+7 (bit 28-31), two's-complement 4-bit (``value & 0xF``).
    Matches ``npu_convert_weight_to_int4pack`` output (int32 [k, n//8] 交叠排放, 对应
    ``npu_weight_quant_batchmatmul`` 的 weight 输入; 区别于 ``pack_int4`` 的 int8
    nibble [k//2, n] 给 ``npu_quant_matmul`` x2).

    注: 交叠排列为 8 int4 连续填 int32 低->高 4 位 (n=8i -> bit0-3, n=8i+7 -> bit28-31).
    已对齐 ops-nn canonical unpack_int32_to_int4 (matmul_quant_util.py) bit-exact 验证 (含 -8).
    CANN 实际排列在 libop_plugin.so (C++, 本地无源码), 与上述一致 (经 ops-nn golden 交叉验证).
    """
    if q.shape[-1] % 8 != 0:
        raise ValueError(f"last dim {q.shape[-1]} must be multiple of 8 for int4pack")
    q = (q.to(torch.int8) & 0x0F).to(torch.int32)
    packed = (
        q[..., 0::8]
        | (q[..., 1::8] << 4)
        | (q[..., 2::8] << 8)
        | (q[..., 3::8] << 12)
        | (q[..., 4::8] << 16)
        | (q[..., 5::8] << 20)
        | (q[..., 6::8] << 24)
        | (q[..., 7::8] << 28)
    )
    return packed.to(torch.int32)


def materialize_int4_per_group_symmetric(
    weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    signed: bool = True,
    nbits: int = 4,
    packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear symmetric quant: q = clamp(round(w/s), ±qmax); dequant w ≈ q*s (offset=0).

    Args:
        weight: ``[out, in]`` (fp16/fp32, GPTQ-corrected).
        scale: ``[out, in//group]`` (fp32, per output-channel × per K-group) — e.g.
            the dopt ``quant_op.weight_quantizer.s`` reshaped from its flattened
            ``[out * in//group]`` form back to 2D.
        group_size: K-axis group size (must match the scale's grouping).
        signed: True -> signed int4 ``[-8, 7]``; False -> unsigned ``[0, 15]``.
            Both dequant as q*s (symmetric, offset=0); unsigned only shifts the
            int range, introducing no zero-point.
        nbits: 4 (int4) default.
        packed: ``True`` -> nibble-packed int8 ``[out, in//2]`` for
            ``cann_lm::quant_matmul`` x2; ``"int4pack"`` -> CANN int4pack int32
            ``[in, out//8]`` (=[K, N//8]) for ``npu_weight_quant_batchmatmul``
            (npu_convert_weight_to_int4pack 格式, weight 需转置到 [K,N] 再沿 N 维 pack);
            ``False`` (default) -> unpack ``[out, in]``.

    Returns:
        (weight_q: int8 ``[out, in]`` (or ``[out, in//2]`` packed), scale: fp32 ``[out, in//group]``).
        Symmetric (offset=0). The scale is returned as-is (already GPTQ-optimized).
    """
    w = weight.detach().float()
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(
            f"in_features ({in_f}) not divisible by group_size ({group_size})"
        )
    if scale.shape != (out_f, in_f // group_size):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} != ({out_f}, {in_f // group_size})"
        )
    qmax = (1 << (nbits - 1)) - 1  # signed positive max: 7 for 4-bit
    qmin = -(
        1 << (nbits - 1)
    )  # signed negative min: -8 (two's-complement full range, matches dopt qmin_max)
    scale_exp = scale.float().repeat_interleave(group_size, dim=-1)  # [out, in]
    q = torch.round(w / scale_exp)
    if signed:
        q = q.clamp(qmin, qmax)  # [-8, 7] standard signed int4 (was [-7,7], dropped -8)
    else:
        q = q.clamp(0, 2 * qmax + 1)
    q = q.to(torch.int8)
    if packed == "int4pack":
        # CANN int4pack: weight [out,in]=[N,K] -> 转置 [in,out]=[K,N] (CANN [k,n]) -> int32 [K, N//8]
        payload = pack_int4_int32(q.contiguous())  # [in, out//8] = [K, N//8] int32
    elif packed:
        payload = pack_int4(q)  # [out, in//2] int8 nibble (npu_quant_matmul x2)
    else:
        payload = q  # [out, in] int8 unpacked
    return payload, scale.float()


def materialize_int8_per_channel_symmetric(
    weight: torch.Tensor,
    scale: torch.Tensor,
    signed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize an embedding weight into (int8 signed, per-channel scale).

    8-bit per-row (per-vocab-channel) symmetric quantization, matching dopt
    ``Quant_Embed_MinMax`` (weight_algo=min_max, bit=8, per_channel=True, sym).

    Args:
        weight: ``[vocab, hidden]`` (fp, dopt fakequant = q*scale stored as float).
        scale: ``[vocab]`` (fp32, per-vocab-row scale from dopt
            ``quant_op.weight_quantizer.s``). dopt offset=0 (sym).
        signed: True -> signed ``[-128, 127]`` (dequant = q*scale, no zero-point).

    Returns:
        (weight_q: int8 ``[vocab, hidden]``, scale: fp32 ``[vocab]``).
        dopt weight is fakequant = q*scale, so round(w/scale[:,None]) = q exactly
        (maxdiff=0 reconstruction); clamp guards against fp drift.
    """
    w = weight.detach().float()
    vocab, hidden = w.shape
    if scale.shape != (vocab,):
        raise ValueError(f"scale shape {tuple(scale.shape)} != ({vocab},)")
    # signed int8 range [-128, 127]; dopt embed uses offset=0 sym, q can hit -128
    qmin, qmax = (-128, 127) if signed else (0, 255)
    s = scale.float().unsqueeze(1)  # [vocab, 1] broadcast over hidden
    q = torch.round(w / s)
    q = q.clamp(qmin, qmax)
    return q.to(torch.int8), scale.float()
