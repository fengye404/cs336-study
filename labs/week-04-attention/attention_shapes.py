from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        # 每个 head 分到同样大小的向量，所以 channels 必须能整除 num_heads。
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        # 一次线性变换同时算出 q/k/v，再用 chunk 切开。
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, time, channels)
        batch, time, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 变成多头格式：(batch, heads, time, head_dim)。
        # transpose 后，heads 维度提前，方便每个 head 独立做 attention。
        q = q.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

        # scores: (batch, heads, time, time)，表示每个 token 看每个 token 的分数。
        scores = q @ k.transpose(-2, -1)
        # 除以 sqrt(head_dim) 是为了避免点积随维度变大而数值过大。
        scores = scores / math.sqrt(self.head_dim)

        # 下三角 mask：第 t 个 token 只能看 0..t，不能偷看未来 token。
        causal_mask = torch.tril(torch.ones(time, time, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        # 用 attention 权重对 v 加权求和，再把多个 head 拼回 channels。
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        out = self.proj(out)
        return out, weights


def main() -> None:
    torch.manual_seed(0)
    # 这里故意用很小的 shape，方便把 attention 矩阵直接打印出来看。
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
