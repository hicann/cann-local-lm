from __future__ import annotations

import torch
from torch import nn, Tensor


class SimpleNetwork(nn.Module):
    """A minimal network using only add, mul, matmul, and rmsnorm operators.

    Architecture:
        x → rmsnorm → matmul(w1) → mul(gate) → matmul(w2) → add(bias) → add(residual) → out
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.w1 = nn.Parameter(torch.randn(dim, dim * 2))
        self.w2 = nn.Parameter(torch.randn(dim * 2, dim))
        self.gate = nn.Parameter(torch.randn(dim * 2))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x)
        h = torch.matmul(h, self.w1)
        h = h * self.gate
        h = torch.matmul(h, self.w2)
        h = h + self.bias
        return x + h


def main() -> None:
    dim = 64
    model = SimpleNetwork(dim).eval()
    x = torch.randn(2, dim)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
