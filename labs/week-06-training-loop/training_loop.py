from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


TEXT = (
    "small experiments should be reproducible. "
    "training loss tells one story and validation loss tells another. "
    "checkpoints let training continue after interruption. "
) * 50


@dataclass
class Config:
    block_size: int = 24
    batch_size: int = 32
    embed_dim: int = 32
    steps: int = 200
    lr: float = 3e-3
    grad_clip: float = 1.0


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, vocab_size),
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.net(self.embedding(idx))
        loss = None
        if targets is not None:
            batch, time, vocab = logits.shape
            loss = F.cross_entropy(logits.view(batch * time, vocab), targets.view(batch * time))
        return logits, loss


def make_dataset():
    chars = sorted(set(TEXT))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in TEXT], dtype=torch.long)
    split = int(0.9 * len(data))
    return data[:split], data[split:], stoi


def get_batch(data: torch.Tensor, config: Config):
    starts = torch.randint(0, len(data) - config.block_size - 1, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(model: TinyLM, train_data: torch.Tensor, val_data: torch.Tensor, config: Config):
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        values = []
        for _ in range(10):
            x, y = get_batch(data, config)
            _, loss = model(x, y)
            assert loss is not None
            values.append(loss.item())
        losses[name] = sum(values) / len(values)
    model.train()
    return losses


def main() -> None:
    torch.manual_seed(2026)
    config = Config()
    train_data, val_data, stoi = make_dataset()
    model = TinyLM(vocab_size=len(stoi), config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)

    checkpoint_dir = Path("checkpoints/generated")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "week-06-tiny-lm.pt"

    print(f"config: {asdict(config)}")
    print(f"vocab_size: {len(stoi)}")

    for step in range(1, config.steps + 1):
        x, y = get_batch(train_data, config)
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 50 == 0:
            losses = estimate_loss(model, train_data, val_data, config)
            print(
                f"step {step:04d} | "
                f"train={losses['train']:.4f} | "
                f"val={losses['val']:.4f} | "
                f"grad_norm={float(grad_norm):.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

    torch.save(
        {
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_path,
    )
    print(f"saved checkpoint: {checkpoint_path}")

    loaded = torch.load(checkpoint_path, weights_only=False)
    print(f"loaded checkpoint keys: {sorted(loaded.keys())}")


if __name__ == "__main__":
    main()

