from __future__ import annotations

import math
import random

import torch
from torch import nn


def make_dataset(n: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-2 * math.pi, 2 * math.pi, n).unsqueeze(1)
    noise = 0.05 * torch.randn_like(x)
    y = torch.sin(x) + 0.3 * torch.cos(3 * x) + noise
    return x, y


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)

    x, y = make_dataset()
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()

    print("shape check")
    print(f"x_train: {tuple(x_train.shape)}")
    print(f"y_train: {tuple(y_train.shape)}")
    print(f"prediction: {tuple(model(x_train[:8]).shape)}")
    print()

    batch_size = 128
    for step in range(1, 801):
        idx = torch.randint(0, len(x_train), (batch_size,))
        pred = model(x_train[idx])
        loss = loss_fn(pred, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0:
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val)
            print(
                f"step {step:04d} | "
                f"train_loss={loss.item():.5f} | "
                f"val_loss={val_loss.item():.5f}"
            )

    with torch.no_grad():
        sample_x = torch.tensor([[-3.0], [0.0], [3.0]])
        sample_y = model(sample_x)

    print()
    print("sample predictions")
    for value, pred in zip(sample_x.squeeze().tolist(), sample_y.squeeze().tolist()):
        print(f"x={value:+.1f} -> y_hat={pred:+.4f}")


if __name__ == "__main__":
    main()

