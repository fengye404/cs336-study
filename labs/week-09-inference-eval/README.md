# Week 09：Inference、Sampling 和 Evaluation

目标：把 logits、文本生成和简单 evaluation 连起来。

## 这个 Lab 放在哪里

对应 lecture：

- inference、evaluation，以及 data 部分的一小块。

对应官方作业：

- Assignment 2：inference/system benchmarking 概念。
- Assignment 4：Data 和 evaluation，但这个 lab 只覆盖很小一块。

做这个 lab 前：

- 完成 Labs 03-06。
- 理解 logits 和 cross entropy。

做完这个 lab 后：

- 做 Assignment 2 时，把 sampling/inference 和 runtime constraints 联系起来。
- 做 Assignment 4 前，要看官方 data lectures；这个 repo 不能替代 data-cleaning 作业。

运行：

```bash
source .venv/bin/activate
python labs/week-09-inference-eval/sampling_and_eval.py
```

重点观察：

- greedy decoding 总是选概率最高的 token。
- temperature 会改变分布的“自信程度”。
- top-k 和 top-p 会限制候选 token 集合。
- perplexity 是平均 negative log-likelihood 的指数形式。

问题：

- 为什么低 temperature 往往让 agent 行为更稳定？
- 为什么 top-p 有时比 top-k 更灵活？
- 为什么一个 eval set 永远不够？
