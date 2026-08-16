"""Export single-op quant_matmul with PROTO-ALIGNED config (monkey-patch, no cann_lm change).

Tests whether x2=DT_INT4 (x2_dtype=16, torch.quint4x2) + transpose_x2=True
(proto: "should always be true") hits the prebuilt ND_ND_int8_int4 kernel
(tilingKey=24) on OM compile, avoiding the V4→V3 fusion that fails the baseline.

Monkey-patches _quant_matmul_converter.convert_quant_matmul to RESPECT the
transpose_x2 param (original hardcodes False + dels it). No cann_lm/ change.

Layout (proto, transpose_x2=True):
  weight_q    : int8 [n, k]   (int4 values unpack; tagged DT_INT4 via x2_dtype=16)
  weight_scale : fp32 [n, k//group]  (NOT pre-transposed)
  x1 (act)    : int8 [m, k]   (runtime input)
  pertoken     : fp32 [m]     (runtime input)
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import List, Optional

import torch
from torch import nn
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.ge_converter.converter_utils import GROUP_SIZE_MAX_VALUE
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import (
    Tensor,
    TensorSpec,
    torch_dtype_value_to_ge_type,
    torch_dtype_value_to_ge_proto_type,
)

from cann_lm.converter.export.epair_utils import consolidated_weights_save
from cann_lm.converter.export.ge_converter import register_ge_converters
import torch_delegate

# registers npu_quant_matmul (schema + fake + cpu impl) on stub env
from cann_lm.models.fused_ops import ops as _fused_ops  # noqa: F401


# ── 1. Monkey-patch converter: respect transpose_x2 (original hardcodes False) ──
def convert_quant_matmul_proto(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    *,
    offset: Optional[Tensor] = None,
    pertoken_scale: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    output_dtype: int = None,
    x1_dtype: int = None,
    x2_dtype: int = None,
    pertoken_scale_dtype: int = None,
    scale_dtype: int = None,
    group_sizes: Optional[List[int]] = None,
    y_scale: Optional[Tensor] = None,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    meta_outputs: TensorSpec = None,
):
    """cann_lm::quant_matmul -> ge.QuantBatchMatmulV4 (PROTO-ALIGNED, respects transpose_x2)."""
    del meta_outputs
    if output_dtype is None:
        output_dtype = 1
    dtype = torch_dtype_value_to_ge_type(output_dtype)

    if x1_dtype is not None and x1 is not None:
        x1 = ge.Bitcast(x1, type=torch_dtype_value_to_ge_type(x1_dtype))
        x1.desc.dtype = torch_dtype_value_to_ge_proto_type(x1_dtype)
    if x2_dtype is not None and x2 is not None:
        x2 = ge.Bitcast(x2, type=torch_dtype_value_to_ge_type(x2_dtype), keep_dim=True)
        x2.desc.dtype = torch_dtype_value_to_ge_proto_type(x2_dtype)
    if pertoken_scale is not None and pertoken_scale_dtype is not None:
        pertoken_scale = ge.Bitcast(
            pertoken_scale, type=torch_dtype_value_to_ge_type(pertoken_scale_dtype))
        pertoken_scale.desc.dtype = torch_dtype_value_to_ge_proto_type(pertoken_scale_dtype)
    if scale_dtype is not None:
        scale = ge.Bitcast(scale, type=torch_dtype_value_to_ge_type(scale_dtype))
        scale.desc.dtype = torch_dtype_value_to_ge_proto_type(scale_dtype)

    group_size = 0
    if group_sizes is not None and isinstance(group_sizes, List):
        if len(group_sizes) != 3:
            raise RuntimeError(
                "group_size must be a list with 3 elements, actual group_sizes is " + str(group_sizes))
        group_m, group_n, group_k = group_sizes
        group_max = GROUP_SIZE_MAX_VALUE
        if group_m > group_max or group_n > group_max or group_k > group_max:
            raise RuntimeError(
                "group_size can't larger than 65535, actual group_sizes is " + str(group_sizes))
        if group_m < 0 or group_n < 0 or group_k < 0:
            raise RuntimeError(
                "group_size can't smaller than 0, actual group_sizes is " + str(group_sizes))
        group_size = (group_m << 32) + (group_n << 16) + group_k

    if bias is None:
        n = x2.shape[-1] if hasattr(x2, 'shape') else None
        bias = ge.Const([0.0] * int(n), dtype=1) if n else None

    # KEY DIFFERENCE vs original: respect transpose_x1/transpose_x2 (not hardcoded False)
    out = ge.QuantBatchMatmulV4(
        x1,
        x2,
        bias=bias,
        x1_scale=pertoken_scale,
        x2_scale=scale,
        y_scale=y_scale,
        x1_offset=None,
        x2_offset=offset,
        y_offset=None,
        x2_table=None,
        dtype=dtype,
        transpose_x1=transpose_x1,
        transpose_x2=transpose_x2,
        group_size=group_size,
    )
    out.desc.dtype = torch_dtype_value_to_ge_proto_type(output_dtype)
    return out


register_ge_converters()  # register all (including original npu_quant_matmul converter)
# OVERRIDE npu_quant_matmul converter with proto-aligned version (last registration wins)
register_fx_node_ge_converter(torch.ops.npu.npu_quant_matmul.default)(convert_quant_matmul_proto)


# ── 2. Module: proto config (transpose_x2=True, x2_dtype=16, PACKED int4) ──
def _pack_int4_2per_byte(wq_int8: torch.Tensor) -> torch.Tensor:
    """Pack [n, k] int8 (int4 values [-8,7], 1/int8) -> [n, k//2] int8 (2 int4/byte).

    Needed because DT_INT4 storage is 4-bit; unpack int8 (1 int4/byte) Bitcast to
    int4 doubles the element count (k -> 2k, wrong). Packed 2/byte -> Bitcast keeps k.
    Layout: low nibble = even-index value, high nibble = odd-index value (order TBD
    vs CANN int4pack; data correctness is a later concern, this is for OM compile test).
    """
    n, k = wq_int8.shape
    assert k % 2 == 0, f"k ({k}) must be even for 2-int4/byte packing"
    pair = wq_int8.view(n, k // 2, 2)
    packed = (pair[:, :, 0] & 0xF) | ((pair[:, :, 1] & 0xF) << 4)
    return packed.to(torch.int8)


class QuantMatmulSingleOpProto(nn.Module):
    def __init__(self, m: int = 4, k: int = 256, n: int = 8, group_size: int = 256):
        super().__init__()
        self.group_size = group_size
        self.n_out = n
        # proto config: packed int8 [n,k//2] + Bitcast(int8->DT_INT4, keep_dim=True) -> [n,k] int4
        wq_unpacked = torch.randint(-8, 7, (n, k), dtype=torch.int8)
        self.register_buffer("weight_q", _pack_int4_2per_byte(wq_unpacked))  # [n, k//2] int8 packed
        # per-channel × per-K-group scale, native [n, k//group] (NOT pre-transposed)
        sc = torch.rand(n, k // group_size, dtype=torch.float32) * 0.01 + 1e-3
        self.register_buffer("weight_scale", sc)
        # dummy zeros bias as a Const buffer (avoids aten.zeros -> Cast node in forward,
        # whose output dtype isn't set by torchair -> DT_INVALID -> no Cast kernel)
        self.register_buffer("dummy_bias", torch.zeros(n, dtype=torch.float16))

    def forward(self, x1: torch.Tensor, pertoken_scale: torch.Tensor) -> torch.Tensor:
        # proto config: x2_dtype=16 (Bitcast int8->DT_INT4) + transpose_x2=True
        return torch.ops.npu.npu_quant_matmul(
            x1, self.weight_q, self.weight_scale,
            pertoken_scale=pertoken_scale,
            bias=self.dummy_bias,
            output_dtype=5,        # 5 = fp16
            x2_dtype=16,           # 16 = torch.quint4x2 -> DT_INT4 (Bitcast packed int8 -> int4)
            group_sizes=[1, 1, self.group_size],
            transpose_x1=False,
            transpose_x2=True,     # proto: "should always be true"
        )


ROOT = Path(__file__).resolve().parent

torch.manual_seed(43)
m, k, n, gs = 4, 256, 8, 256
net = QuantMatmulSingleOpProto(m, k, n, gs).eval()
x1 = torch.randint(-127, 127, (m, k), dtype=torch.int8)
pertoken = torch.ones(m, dtype=torch.float32) * 0.01
sample_inputs = (x1, pertoken)

exported_program = torch.export.export(net, sample_inputs, strict=False)
epair = torch_delegate.export_to_epair(exported_program)

output_dir = ROOT / "epair_model_proto"
consolidated_weights_save(epair, output_dir, save_weights=True)

for i, inp in enumerate(sample_inputs):
    (ROOT / f"input_proto_{i}.bin").write_bytes(inp.cpu().contiguous().numpy().tobytes())
    print(f"Saved input_proto[{i}] dtype={inp.dtype} shape={list(inp.shape)}")

print(f"\nepair (proto config) saved to: {output_dir}")
print(f"  config: x2=DT_INT4 (x2_dtype=16), transpose_x2=True, out=fp16")
print(f"\nNext: OM-compile on NPU — expect to hit ND_ND_int8_int4_fp16 (tilingKey=24)")
