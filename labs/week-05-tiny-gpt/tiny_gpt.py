from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


TEXT = (
    "agents plan actions observe results and update context. "
    "language models predict tokens from previous tokens. "
    "attention lets each token read useful earlier tokens. "
) * 40


class CausalSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:time, :time] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, channels: int, num_heads: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(channels)
        self.attn = CausalSelfAttention(channels, num_heads, block_size)
        self.ln2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int) -> None:
        super().__init__()
        channels = 64
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, channels)
        self.position_embedding = nn.Embedding(block_size, channels)
        self.blocks = nn.Sequential(
            Block(channels, num_heads=4, block_size=block_size),
            Block(channels, num_heads=4, block_size=block_size),
        )
        self.ln = nn.LayerNorm(channels)
        self.lm_head = nn.Linear(channels, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch, time = idx.shape
        positions = torch.arange(time, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch * time, -1), targets.view(batch * time))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def make_data():
    chars = sorted(set(TEXT))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in TEXT], dtype=torch.long)
    return data, stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def decode(ids: torch.Tensor, itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids.tolist())


def main() -> None:
    torch.manual_seed(123)
    block_size = 32
    data, stoi, itos = make_data()
    model = TinyGPT(vocab_size=len(stoi), block_size=block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    x, y = get_batch(data, batch_size=4, block_size=block_size)
    logits, loss = model(x, y)
    assert loss is not None
    print("shape check")
    print(f"x: {tuple(x.shape)}")
    print(f"logits: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
    print()

    for step in range(1, 301):
        x, y = get_batch(data, batch_size=32, block_size=block_size)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f"step {step:04d} | loss={loss.item():.4f}")

    start = torch.tensor([[stoi["a"]]], dtype=torch.long)
    sample = model.generate(start, steps=180)[0]
    print()
    print(decode(sample, itos))


if __name__ == "__main__":
    main()

