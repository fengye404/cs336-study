# Week 02：Tokenization 和 BPE

目标：理解文本如何变成 token ids。

## 这个 Lab 放在哪里

对应 lecture：

- CS336 overview 和 tokenization 相关内容。

对应官方作业：

- Assignment 1: Basics 里的 tokenizer 部分热身。

做这个 lab 前：

- 完成 Lab 01。
- 快速扫一遍 Assignment 1 的 tokenizer 部分。

做完这个 lab 后：

- 继续 Lab 03。
- 可以先读 Assignment 1 的 tokenizer tests，但建议等到 Lab 05 或 Lab 06 后再认真完成整个 Assignment 1。

运行：

```bash
source .venv/bin/activate
python labs/week-02-tokenizer/toy_bpe.py
python labs/week-02-tokenizer/token_count.py
```

重点观察：

- BPE 从小片段开始，不断合并高频相邻 pair。
- 训练 tokenizer 的结果是得到 merge rules。
- encoding 文本时，是在应用已经学到的 merge rules。
- 文案、标点、空格、语言变化都会影响 token 数。

问题：

- vocabulary 是什么？
- merge rule 是什么？
- 为什么 `"hello world"` 不一定就是两个 token？
- tokenization 为什么会影响 agent 的成本和上下文长度？
