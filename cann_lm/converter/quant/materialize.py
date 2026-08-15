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

"""Weight int4 materialization (W4A8 s4, symmetric, per-group, unpack placeholder).

Converts a fp16/bf16 nn.Linear weight [out, in] into a quantized int8 payload
(int4 signed values stored 1/byte = unpack) + per-group fp32 scale along `in`
(K dim). offset = 0 (symmetric).

遗留: int4 编码 (signed [-8,7] vs unsigned [0,15])、packed (2/byte) vs unpack
(1/byte)、scale 形状是 CANN QuantBatchMatmulV4 约定缺口（见
docs/cann_lm_converter/quant_design.md "int4 物化的 CANN 约定"）。本模块为
unpack+signed 占位实现，先跑通 Python 流程，CANN 对齐后续。
"""

import torch


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Nibble-pack signed int4 values (stored as int8) into [n, k//2] int8.

    Input: ``q`` int8 ``[..., k]`` holding signed int4 values in [-8, 7].
    Output: int8 ``[..., k//2]`` — byte at index i holds int4 for k=2i (low
    nibble) and k=2i+1 (high nibble), two's-complement 4-bit (``value & 0xF``).
    Matches the unpack in ``fused_ops/quant_matmul_s8s4._quant_matmul_s8s4_cpu``
    and the op ``x2`` input layout (方案 A).
    """
    if q.shape[-1] % 2 != 0:
        raise ValueError(f"last dim {q.shape[-1]} must be even for int4 nibble-pack")
    q = q.to(torch.int8)
    nib_even = q[..., 0::2] & 0x0F
    nib_odd = q[..., 1::2] & 0x0F
    return ((nib_odd << 4) | nib_even).to(torch.int8)


def materialize_int4(
    weight: torch.Tensor,
    group_size: int = 128,
    signed: bool = True,
    nbits: int = 4,
    packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a Linear weight into (int4-as-int8 payload, per-group scale).

    Args:
        weight: fp16/bf16/fp32 weight in nn.Linear orientation ``[out, in]``.
        group_size: K-axis (in) group size; ``in`` must be divisible by it.
        signed: True -> signed symmetric range ``[-qmax, qmax]`` (qmax=7 for 4-bit);
            False -> unsigned ``[0, 2*qmax+1]``.
        nbits: bits per weight element (4 for int4).
        packed: True -> return nibble-packed int8 ``[out, in//2]`` (2 int4/byte,
            低 nibble=偶 in 索引) for the 方案 A ``quant_matmul_s8s4`` op x2 input;
            False (default) -> unpack int8 ``[out, in]`` (1 int4/byte).

    Returns:
        (weight_q: int8 ``[out, in]`` (or ``[out, in//2]`` packed), scale: fp32 ``[out, in//group]``).
        Symmetric (offset=0), so no offset tensor.
    """
    w = weight.detach().float()
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(
            f"in_features ({in_f}) not divisible by group_size ({group_size})"
        )
    qmax = (1 << (nbits - 1)) - 1  # signed 4-bit -> 7
    ng = in_f // group_size
    wg = w.reshape(out_f, ng, group_size)
    scale = (wg.abs().amax(dim=-1) / qmax).clamp(min=1e-8)  # [out, ng]
    scale_exp = scale.repeat_interleave(group_size, dim=-1)  # [out, in]
    q = torch.round(w / scale_exp)
    if signed:
        q = q.clamp(-qmax, qmax)
    else:
        q = q.clamp(0, 2 * qmax + 1)
    q = q.to(torch.int8)
    return (pack_int4(q) if packed else q), scale.to(torch.float32)


def materialize_int4_with_scale(
    weight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    signed: bool = True,
    nbits: int = 4,
    packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize using a PRECOMPUTED per-group scale (e.g., from dopt GPTQ).

    Args:
        weight: ``[out, in]`` (fp16/fp32, GPTQ-corrected).
        scale: ``[out, in//group]`` (fp32, per output-channel × per K-group) — e.g.
            the dopt ``quant_op.weight_quantizer.s`` reshaped from its flattened
            ``[out * in//group]`` form back to 2D.
        group_size: K-axis group size (must match the scale's grouping).
        packed: True -> return nibble-packed int8 ``[out, in//2]`` for 方案 A
            ``quant_matmul_s8s4`` x2; False (default) -> unpack ``[out, in]``.

    Returns:
        (weight_q: int8 ``[out, in]`` (or ``[out, in//2]`` packed), scale: fp32 ``[out, in//group]``).
        Symmetric (offset=0). The scale is returned as-is (already GPTQ-optimized).
    """
    w = weight.detach().float()
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features ({in_f}) not divisible by group_size ({group_size})")
    if scale.shape != (out_f, in_f // group_size):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} != ({out_f}, {in_f // group_size})"
        )
    qmax = (1 << (nbits - 1)) - 1  # signed 4-bit -> 7
    scale_exp = scale.float().repeat_interleave(group_size, dim=-1)  # [out, in]
    q = torch.round(w / scale_exp)
    if signed:
        q = q.clamp(-qmax, qmax)
    else:
        q = q.clamp(0, 2 * qmax + 1)
    q = q.to(torch.int8)
    return (pack_int4(q) if packed else q), scale.float()


def materialize_int8_per_channel(
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
