from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


TEXT = """
language models predict the next token
tokens become ids
ids become embeddings
embeddings become logits
logits become probabilities
""".strip()


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 16) -> None:
        super().__init__()
        # Embedding 表可以理解成一个查表：token id -> 向量。
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # lm_head 把每个位置的向量映射成 vocab_size 个 logits。
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: (batch, time)
        # x: (batch, time, embed_dim)
        x = self.token_embedding(idx)
        # logits: (batch, time, vocab_size)，每个位置都预测下一个 token。
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch, time, vocab_size = logits.shape
            # cross_entropy 期望输入是 (N, C)，target 是 (N,)。
            # 所以把 batch 和 time 合并成一个维度。
            loss = F.cross_entropy(
                logits.view(batch * time, vocab_size),
                targets.view(batch * time),
            )
        return logits, loss


def build_data() -> tuple[torch.Tensor, dict[str, int], dict[int, str]]:
    # 字符级 tokenizer：每个不同字符一个 id。
    chars = sorted(set(TEXT))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    # data 是整段文本的 token id 序列。
    data = torch.tensor([stoi[ch] for ch in TEXT], dtype=torch.long)
    return data, stoi, itos


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    # 随机挑 batch_size 个起点，每个样本长度是 block_size。
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    # 语言模型训练目标：用当前位置预测下一个 token。
    # 所以 y 是 x 整体向后移动一位。
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def generate(model: BigramLanguageModel, start: int, steps: int, itos: dict[int, str]) -> str:
    idx = torch.tensor([[start]], dtype=torch.long)
    for _ in range(steps):
        # bigram 模型只看最后一个 token 来预测下一个 token。
        logits, _ = model(idx[:, -1:])
        # softmax 把 logits 转成概率分布。
        probs = F.softmax(logits[:, -1, :], dim=-1)
        # multinomial 按概率采样，而不是永远选最大值。
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())


def main() -> None:
    torch.manual_seed(7)
    data, stoi, itos = build_data()
    model = BigramLanguageModel(vocab_size=len(stoi))
    # 这里 lr 偏大一点，是为了让小实验快速看到 loss 下降。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x, y = get_batch(data, batch_size=4, block_size=8)
    logits, loss = model(x, y)

    print("shape check")
    print(f"input ids x: {tuple(x.shape)}")
    print(f"target ids y: {tuple(y.shape)}")
    print(f"logits: {tuple(logits.shape)}")
    print(f"loss: {loss.item():.4f}")
    print()

    for step in range(1, 501):
        x, y = get_batch(data, batch_size=32, block_size=16)
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step {step:04d} | loss={loss.item():.4f}")

    print()
    print("sample")
    print(generate(model, start=stoi["l"], steps=120, itos=itos))


if __name__ == "__main__":
    main()
