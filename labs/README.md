# Labs

这里放小型可运行实验。这些是我们自己的学习 lab，不是 CS336 官方作业，也不是官方答案。

## 和 CS336 的关系

CS336 的学习材料可以分成三层：

- Lecture：官方讲概念、背景和设计取舍。
- 我们自己的 lab：把单个概念拆成能跑的小练习。
- 官方 assignment：更大的综合实现，脚手架少，难度高。

推荐顺序：

1. 先看或快速扫一遍相关 lecture。
2. 再跑对应 lab，把概念落到代码和 shape 上。
3. 等几个相关 lab 都比较熟了，再开始官方 assignment。

这些 lab 会比官方作业小很多，也会更有引导。lab 里可以直接用 PyTorch；官方作业则可能要求你自己实现更底层的东西。

从仓库根目录运行：

```bash
source .venv/bin/activate
python labs/week-01-pytorch-basics/train_tiny_mlp.py
```

## Lab 地图

| Week | Lab | 用途 |
| --- | --- | --- |
| 01 | `week-01-pytorch-basics` | PyTorch 训练循环 |
| 02 | `week-02-tokenizer` | 玩具版 BPE 和 token 计数 |
| 03 | `week-03-lm-objective` | embedding、logits、cross entropy |
| 04 | `week-04-attention` | causal self-attention 的 shape |
| 05 | `week-05-tiny-gpt` | 最小 GPT 风格模型 |
| 06 | `week-06-training-loop` | validation、checkpoint、gradient clipping |
| 07 | `week-07-memory-estimates` | 粗略估算显存 |
| 08 | `week-08-training-systems` | 分布式训练概念地图 |
| 09 | `week-09-inference-eval` | sampling 和 perplexity |
| 10 | `week-10-alignment-bridge` | preference data 和 DPO loss |

每个 lab 最好产出一份 note 或 checkpoint。重点不是把代码写漂亮，而是把概念变成自己能运行、能解释的东西。

## 官方作业时间点

| 官方任务 | 做完哪些 lab 后开始 | 为什么 |
| --- | --- | --- |
| Assignment 1: Basics | Labs 02-06 | tokenizer、模型结构、objective、attention、optimizer/training loop 都会用到 |
| Assignment 2: Systems | Labs 04、06、07、08、09 | attention、benchmark 思维、显存估算、parallelism、inference |
| Assignment 3: Scaling | Labs 06-08 | 实验记录、显存/FLOP 估算、scaling 直觉 |
| Assignment 4: Data | Lab 09 + CS336 data lectures | 本仓库只轻量覆盖 eval/data，data 部分要重看官方 Lecture 13-14 |
| Assignment 5: Alignment | Lab 10 | SFT、preference data、RLHF/DPO 桥接 |

快攻规则：做完 Lab 05 后认真读 Assignment 1；做完 Lab 06 后正式开始实现 Assignment 1。

## Lecture 到 Lab 路线

不知道下一步学什么时，看这张表。

| 学习阶段 | 先看什么 | 再跑什么 | 然后尝试什么 |
| --- | --- | --- | --- |
| 热身 | CS336 prerequisites 和 Lecture 1 overview | Lab 01 | 暂时不做官方作业 |
| Tokenization | Lecture 1 的 tokenization 部分 | Lab 02 | Assignment 1 tokenizer tests |
| LM objective | Transformer / language modeling 相关材料 | Lab 03 | Assignment 1 model/loss code |
| Attention | Transformer attention 相关材料 | Lab 04 | Assignment 1 attention/model tests |
| Tiny GPT | Transformer architecture 相关材料 | Lab 05 | Assignment 1 full model implementation |
| Training | optimization / training loop 相关材料 | Lab 06 | Assignment 1 training 和 optimizer 部分 |
| Resource accounting | Lecture 2 风格的 PyTorch/resource accounting 内容 | Lab 07 | Assignment 2 和 Assignment 3 handouts |
| Systems | Lecture 7 parallelism 和 systems 内容 | Lab 08 | Assignment 2 implementation |
| Inference/eval/data | Lecture 10 inference、Lecture 12 evaluation、Lecture 13-14 data | Lab 09 | Assignment 4，前提是 data lectures 已看 |
| Alignment | alignment / RLHF 相关材料 | Lab 10 | Assignment 5 |

如果 lecture 太抽象，可以先跑 lab 再回来读。  
如果 lab 像魔法，说明概念没接上，回到 lecture，把缺的那一块写进 notes。
