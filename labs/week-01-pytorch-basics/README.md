# Week 01：PyTorch 基础

目标：跑通并理解一个完整的 PyTorch 训练循环。

## 这个 Lab 放在哪里

CS336 默认你对 Python、PyTorch 和系统基础比较舒服。这个 lab 是我们自己的热身，目的是在官方内容加速前，先把训练循环跑通。

做这个 lab 前：

- 快速看一下 CS336 首页和 prerequisites。
- 读一遍 `study-plan.md` 里的 Week 1。

做完这个 lab 后：

- 继续 Lab 02。
- 如果训练循环还读不顺，先不要正式开始 Assignment 1。

运行：

```bash
source .venv/bin/activate
python labs/week-01-pytorch-basics/train_tiny_mlp.py
```

重点观察：

- 开头打印出来的 tensor shapes。
- training loss 应该下降。
- validation loss 应该在合理范围内。
- 模型应该比随机猜测更能拟合这个合成函数。

在 `notes/week-01-setup.md` 里回答：

- tensor 和 NumPy array 有什么区别？
- `loss.backward()` 算出了什么？
- 为什么反向传播前要调用 `optimizer.zero_grad()`？
- 这个模型里流过了哪些 tensor shapes？
