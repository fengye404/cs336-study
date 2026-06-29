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
    # 把实验参数集中放在 Config 里，方便每次只改一个变量做对比。
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
        # 这是一个极简 LM：每个 token 独立过 MLP，没有 attention。
        # 它主要用来练 training loop，不是为了生成高质量文本。
        logits = self.net(self.embedding(idx))
        loss = None
        if targets is not None:
            batch, time, vocab = logits.shape
            # 把 (batch, time) 展平成一个大 batch 来算 cross entropy。
            loss = F.cross_entropy(logits.view(batch * time, vocab), targets.view(batch * time))
        return logits, loss


def make_dataset():
    # 字符级数据集。真实 LLM 会用 tokenizer，这里为了练训练流程保持简单。
    chars = sorted(set(TEXT))
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in TEXT], dtype=torch.long)
    # 90% 训练，10% 验证。验证集不参与参数更新。
    split = int(0.9 * len(data))
    return data[:split], data[split:], stoi


def get_batch(data: torch.Tensor, config: Config):
    # 每个 batch 随机抽一些连续片段。
    starts = torch.randint(0, len(data) - config.block_size - 1, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in starts])
    # y 永远是 x 向后移动一位。
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(model: TinyLM, train_data: torch.Tensor, val_data: torch.Tensor, config: Config):
    # eval 模式会关闭 dropout 等训练专用行为。这个模型没有 dropout，但习惯要保留。
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        values = []
        for _ in range(10):
            # 多采几次 batch 再平均，避免单个 batch 的 loss 太偶然。
            x, y = get_batch(data, config)
            _, loss = model(x, y)
            assert loss is not None
            values.append(loss.item())
        losses[name] = sum(values) / len(values)
    # 评估完切回 train 模式，继续训练。
    model.train()
    return losses


def main() -> None:
    torch.manual_seed(2026)
    config = Config()
    train_data, val_data, stoi = make_dataset()
    model = TinyLM(vocab_size=len(stoi), config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    # CosineAnnealingLR 会让学习率从初始值逐渐降到接近 0。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)

    # 生成的 checkpoint 放在 checkpoints/generated，已经被 .gitignore 忽略。
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
        # 梯度裁剪：如果梯度过大，把整体范数限制住，避免训练炸掉。
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
            # 完整恢复训练通常要保存：配置、模型参数、optimizer 状态、scheduler 状态。
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
