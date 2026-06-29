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
        # 多头 attention 会把 channels 平均分给每个 head。
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        # qkv 一次性算出 query/key/value，减少三次线性层的样板代码。
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        # register_buffer 注册的是“不是参数、但要跟着模型移动/保存”的 tensor。
        # mask 不需要训练，所以不用 nn.Parameter。
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # 多头 attention 标准 shape: (batch, heads, time, head_dim)。
        q = q.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

        # q @ k^T 得到 token 之间的相似度分数。
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        # causal mask 保证当前位置不能看未来 token。
        scores = scores.masked_fill(self.mask[:time, :time] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        # 把 heads 维度拼回 channels，恢复 (batch, time, channels)。
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
        # Pre-LN Transformer block：先 LayerNorm，再 attention/MLP。
        # 残差连接 x + f(x) 保持 shape 不变，也让梯度更容易流动。
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int) -> None:
        super().__init__()
        channels = 64
        self.block_size = block_size
        # token_embedding 负责“这个 token 是什么”。
        self.token_embedding = nn.Embedding(vocab_size, channels)
        # position_embedding 负责“这个 token 在第几个位置”。
        self.position_embedding = nn.Embedding(block_size, channels)
        self.blocks = nn.Sequential(
            Block(channels, num_heads=4, block_size=block_size),
            Block(channels, num_heads=4, block_size=block_size),
        )
        self.ln = nn.LayerNorm(channels)
        self.lm_head = nn.Linear(channels, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: (batch, time)，里面是 token ids。
        batch, time = idx.shape
        positions = torch.arange(time, device=idx.device)
        # token 信息和位置信息相加，得到每个位置的初始表示。
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.ln(x)
        # logits: (batch, time, vocab_size)，每个位置预测下一个 token。
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # cross_entropy 要求 (N, classes) 和 (N,)，所以合并 batch/time。
            loss = F.cross_entropy(logits.view(batch * time, -1), targets.view(batch * time))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, steps: int) -> torch.Tensor:
        for _ in range(steps):
            # 如果序列超过 block_size，只保留最后 block_size 个 token 作为上下文。
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            # 只取最后一个位置的 logits，因为生成时只需要预测“下一个 token”。
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def make_data():
    # 字符级 tokenizer：简单但足够演示语言模型训练。
    chars = sorted(set(TEXT))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in TEXT], dtype=torch.long)
    return data, stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    # y 是 x 的下一个字符序列，用来做 next-token prediction。
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def decode(ids: torch.Tensor, itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids.tolist())


def main() -> None:
    torch.manual_seed(123)
    # block_size 是模型一次最多能看的上下文长度。
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
        # 标准训练三步：清梯度 -> 反向传播 -> 更新参数。
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
