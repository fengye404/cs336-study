from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        causal_mask = torch.tril(torch.ones(time, time, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        out = self.proj(out)
        return out, weights


def main() -> None:
    torch.manual_seed(0)
    batch, time, channels, heads = 2, 5, 12, 3
    x = torch.randn(batch, time, channels)
    attention = CausalSelfAttention(channels=channels, num_heads=heads)

    out, weights = attention(x)

    print("shape check")
    print(f"x: {tuple(x.shape)}")
    print(f"attention weights: {tuple(weights.shape)}")
    print(f"out: {tuple(out.shape)}")
    print()

    print("causal mask behavior for batch 0, head 0")
    print(weights[0, 0])
    print()
    print("upper triangle should be zero after softmax")
    future_weight_sum = weights[0, 0].triu(diagonal=1).sum()
    print(f"future_weight_sum={future_weight_sum.item():.6f}")


if __name__ == "__main__":
    main()

