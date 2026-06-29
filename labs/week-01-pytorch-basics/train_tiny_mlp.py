from __future__ import annotations

import math
import random

import torch
from torch import nn


def make_dataset(n: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    # x 原本是一维向量，unsqueeze(1) 把它变成 (n, 1)。
    # 神经网络通常按二维 batch 输入理解数据：每一行是一个样本，每一列是一个特征。
    x = torch.linspace(-2 * math.pi, 2 * math.pi, n).unsqueeze(1)
    noise = 0.05 * torch.randn_like(x)
    # 这里造一个有规律但不完全干净的函数，让模型学 sin/cos 的组合。
    y = torch.sin(x) + 0.3 * torch.cos(3 * x) + noise
    return x, y


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # nn.Sequential 会按顺序执行这些层：
        # Linear 做仿射变换，Tanh 提供非线性，否则模型只能学直线。
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
    # 固定随机种子，方便你多次运行时看到接近的结果。
    torch.manual_seed(42)
    random.seed(42)

    x, y = make_dataset()
    # 简单切分：前 80% 做训练集，后 20% 做验证集。
    split = int(0.8 * len(x))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    model = TinyMLP()
    # optimizer 负责根据梯度更新模型参数；lr 是每次更新的步子大小。
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # MSELoss 衡量预测值和真实值的平方误差，适合这个回归任务。
    loss_fn = nn.MSELoss()

    print("shape check")
    print(f"x_train: {tuple(x_train.shape)}")
    print(f"y_train: {tuple(y_train.shape)}")
    print(f"prediction: {tuple(model(x_train[:8]).shape)}")
    print()

    batch_size = 128
    for step in range(1, 801):
        # 随机抽一批样本。idx 的 shape 是 (batch_size,)。
        idx = torch.randint(0, len(x_train), (batch_size,))
        # 前向传播：输入 x，得到预测 pred。
        pred = model(x_train[idx])
        loss = loss_fn(pred, y_train[idx])

        # PyTorch 默认会累积梯度，所以每一步训练前要先清空旧梯度。
        optimizer.zero_grad()
        # 反向传播：从 loss 出发，计算每个参数的梯度。
        loss.backward()
        # 根据梯度真正更新参数。
        optimizer.step()

        if step == 1 or step % 100 == 0:
            # 验证时不需要梯度；no_grad 会省内存，也避免误把验证计算放进计算图。
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val)
            print(
                f"step {step:04d} | "
                f"train_loss={loss.item():.5f} | "
                f"val_loss={val_loss.item():.5f}"
            )

    with torch.no_grad():
        # 拿几个没放进 batch 的点，看模型现在会输出什么。
        sample_x = torch.tensor([[-3.0], [0.0], [3.0]])
        sample_y = model(sample_x)

    print()
    print("sample predictions")
    for value, pred in zip(sample_x.squeeze().tolist(), sample_y.squeeze().tolist()):
        print(f"x={value:+.1f} -> y_hat={pred:+.4f}")


if __name__ == "__main__":
    main()
