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
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.token_embedding(idx)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            batch, time, vocab_size = logits.shape
            loss = F.cross_entropy(
                logits.view(batch * time, vocab_size),
                targets.view(batch * time),
            )
        return logits, loss


def build_data() -> tuple[torch.Tensor, dict[str, int], dict[int, str]]:
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


def generate(model: BigramLanguageModel, start: int, steps: int, itos: dict[int, str]) -> str:
    idx = torch.tensor([[start]], dtype=torch.long)
    for _ in range(steps):
        logits, _ = model(idx[:, -1:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return "".join(itos[i] for i in idx[0].tolist())


def main() -> None:
    torch.manual_seed(7)
    data, stoi, itos = build_data()
    model = BigramLanguageModel(vocab_size=len(stoi))
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

