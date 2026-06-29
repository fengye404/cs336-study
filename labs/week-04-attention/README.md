# Week 04：Causal Self-Attention

目标：实现 attention，并检查每一个 shape。

## 这个 Lab 放在哪里

对应 lecture：

- Transformer architecture、attention、attention alternatives。

对应官方作业：

- Assignment 1：实现 Transformer components。
- Assignment 2：后面会优化 attention，也会遇到 FlashAttention 风格的系统约束。

做这个 lab 前：

- 完成 Lab 03。
- 对 `(batch, time, channels)` 这种 tensor shape 不陌生。

做完这个 lab 后：

- 继续 Lab 05。
- 做 Assignment 2 前再回来复习这个 lab，因为系统优化要先理解 attention 的 memory 和 compute pattern。

运行：

```bash
source .venv/bin/activate
python labs/week-04-attention/attention_shapes.py
```

重点观察：

- self-attention 里的 `Q`、`K`、`V` 都来自同一个输入。
- attention scores 的 shape 是 `(batch, heads, time, time)`。
- causal mask 会阻止 token 看到未来位置。
- attention output 会回到 `(batch, time, channels)`。

问题：

- 为什么 attention scores 要除以 `sqrt(head_dim)`？
- causal language modeling 为什么需要 lower-triangular mask？
- 哪些维度分别是 sequence length、embedding size、number of heads？
