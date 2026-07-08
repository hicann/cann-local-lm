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

"""A minimal Transformer network built from scratch with torch.

Usage:
  python -m cann_lm.converter.export.simple_transformer
"""

from __future__ import annotations

import math

import torch
from torch import nn, Tensor


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)

    def forward(self, x: Tensor, start: int = 0) -> tuple[Tensor, Tensor]:
        seq_len = start + x.size(1)
        return (
            self.sin_cache[start:seq_len].unsqueeze(0).unsqueeze(0),
            self.cos_cache[start:seq_len].unsqueeze(0).unsqueeze(0),
        )


def apply_rotary(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, sin: Tensor, cos: Tensor, mask: Tensor) -> Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, sin, cos)
        k = apply_rotary(k, sin, cos)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: Tensor, sin: Tensor, cos: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), sin, cos, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int, n_layers: int, max_seq_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        B, S = tokens.shape
        x = self.embed(tokens)
        sin, cos = self.rope(x)
        mask = torch.triu(
            torch.full((S, S), float("-inf"), dtype=x.dtype, device=x.device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, sin, cos, mask)
        x = self.norm(x)
        return self.lm_head(x)


def main() -> None:
    vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len = 256, 64, 8, 256, 2, 128
    model = SimpleTransformer(vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len).eval()
    tokens = torch.randint(0, vocab_size, (2, 16))
    with torch.no_grad():
        logits = model(tokens)
    print(f"Input shape:  {tokens.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
