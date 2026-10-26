"""Weight format transform (ND <-> FractNZ/FractZ) + rule resolution.

layout.nd2nz: [M,K] ND -> FractNZ block-major (Stage 1, fp16/bf16/int8).
layout.pack_int4_nz: int4pack int32 [K,N//8] -> FractZ (Stage 2, QuantBatchMatmulV4 x2).
rules.resolve_format: name_pattern + op_type -> format config (load_rules reads shipped json).
"""

from .layout import nd2nz, pack_int4_nz
from .rules import load_rules, resolve_format

__all__ = ["nd2nz", "pack_int4_nz", "load_rules", "resolve_format"]
