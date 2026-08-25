"""cann_lm::l2norm — L2 normalize op (schema + meta + CPU eager).

线性注意力 (GatedDeltaNet) 的 query/key l2norm, 替代纯 torch 的
``x * rsqrt(mean(x²)*Dk + eps)`` 分解 (36 个 Rsqrt+Mul+Mean 基元 → 1 个 L2Normalize).
GE 映射 ``ge.L2Normalize`` (support_ops.md fuse_op l2norm).
"""

import torch

from ._library import CANN_LM_LIB, register_cpu_impl

if CANN_LM_LIB is not None:
    CANN_LM_LIB.define(
        "l2norm(Tensor x, int dim=-1, float eps=1e-6) -> Tensor"
    )


if True:  # register_fake for both stub + real torch_npu (cann_lm op, torch_npu 不提供)
    @torch.library.register_fake("cann_lm::l2norm")
    def _l2norm_meta(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        del dim, eps
        return torch.empty_like(x)

    @register_cpu_impl("cann_lm::l2norm")
    def _l2norm_cpu(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm
