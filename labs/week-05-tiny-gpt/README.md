# Week 05：Tiny GPT

目标：把 embeddings、attention、MLP、residuals 和 LM head 组装起来。

## 这个 Lab 放在哪里

对应 lecture：

- Transformer architecture、hyperparameters、normalization、attention、MLP。

对应官方作业：

- Assignment 1: Basics 里的完整模型结构。

做这个 lab 前：

- 完成 Labs 02-04。
- 大致理解 tokenization、logits、attention 和 causal masking。

做完这个 lab 后：

- 认真读 Assignment 1。
- 如果你能解释这个 lab 的 forward pass，就可以开始实现 Assignment 1。
- Lab 06 可以和 Assignment 1 的训练/debugging 并行推进。

运行：

```bash
source .venv/bin/activate
python labs/week-05-tiny-gpt/tiny_gpt.py
```

重点观察：

- token embeddings 和 positional embeddings 会相加。
- 每个 Transformer block 都保持 `(batch, time, channels)`。
- LM head 把 channels 映射到 vocabulary logits。
- 训练后生成的 sample 应该会稍微变得像样一点。

问题：

- 为什么 residual connections 要保持 shape？
- 模型为什么需要 positional embeddings？
- Week 03 的 bigram model 和这个 tiny GPT 有什么关键区别？
