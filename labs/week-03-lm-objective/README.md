# Week 03：Embeddings、Logits 和 LM Objective

目标：理解 next-token prediction。

## 这个 Lab 放在哪里

对应 lecture：

- language modeling objective、embeddings、logits、cross entropy。

对应官方作业：

- Assignment 1: Basics 里的 model/loss 基础。

做这个 lab 前：

- 完成 Lab 02。
- 知道 token ids 是什么，也知道为什么 targets 是 inputs 向后错一位。

做完这个 lab 后：

- 继续 Lab 04，然后再认真写 Assignment 1 的 model 部分。
- 在 Assignment 1 里找到 logits 和 cross entropy 出现的位置。

运行：

```bash
source .venv/bin/activate
python labs/week-03-lm-objective/train_bigram_lm.py
```

重点观察：

- input 是 shape 为 `(batch, time)` 的 token ids。
- embedding 把 ids 映射成向量。
- 模型输出 logits，shape 是 `(batch, time, vocab_size)`。
- cross entropy 拿 logits 和下一个 token ids 做比较。

问题：

- 为什么 targets 等于 inputs 向后错一位？
- logit 是什么？
- 为什么 cross entropy 需要 class indices，而不是 one-hot vectors？
