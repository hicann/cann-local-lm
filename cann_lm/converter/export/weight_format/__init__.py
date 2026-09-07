"""Weight format transform (ND <-> FractNZ) + rule resolution.

layout.nd2nz: [M,K] ND -> FractNZ block-major (Stage 1, fp16/bf16/int8).
rules.resolve_format: name_pattern + op_type -> format config (load_rules reads shipped json).
"""

from .layout import nd2nz
from .rules import load_rules, resolve_format

__all__ = ["nd2nz", "load_rules", "resolve_format"]
