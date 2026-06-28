# 第 1 讲：总览与 Tokenization

来源：`external/cs336-lectures/lecture_01.py`

这是第一讲的中文文本版，按官方 executable lecture 的结构压缩整理。

如果你只看一份材料，优先看可执行 notebook：

- `notes/lectures/lecture-01-overview-tokenization.zh.ipynb`

这个 Markdown 适合快速复习；notebook 是主学习材料，章节顺序和这里一致，并在对应位置加入了可运行实验。

## 1. 为什么要有这门课

研究者和工程师这些年一直在往更高的抽象层走：

- 2016 年左右，很多研究者会自己实现并训练模型。
- 2018 年左右，很多人下载 BERT 之类的预训练模型，然后 fine-tune。
- 今天，很多人直接 prompt GPT、Claude、Gemini 这类 API 模型。

抽象层提高了效率，但语言模型这层抽象并不严密。你想做真正底层、可靠、可迁移的工作，就必须理解 API 下面那套东西：tokenization、模型结构、训练、系统、数据、评估和 alignment。

这门课的核心方法是：通过亲手构建来理解。

## 2. Frontier Model 不可能在课上复现

最前沿的大模型训练成本太高，个人学习仓库或课堂都不可能完整复现。关键问题不是“我们能不能从零训练 GPT-4”，而是：

> 哪些从小模型里学到的知识，可以迁移到 frontier-scale model？

第一讲把可迁移知识分成三类：

- Mechanics：机制。比如 Transformer、tokenizer、optimizer、并行训练、训练循环怎么工作。
- Mindset：思维方式。把 compute、memory、bandwidth、data 都当成稀缺资源。
- Intuitions：经验直觉。比如什么数据好、什么模型设计好。这类东西跨尺度迁移不一定稳定。

## 3. 主线：效率

CS336 会反复回到一个问题：

> 在固定 compute、memory、communication、data 约束下，怎么训练出最好的模型？

所谓 bitter lesson 不是“只要规模够大，算法不重要”。更准确的理解是：

> 能随规模一起扩展的算法才重要。

规模越大，浪费越贵。所以模型结构、kernel、数据过滤、batching、超参数选择，本质上都是效率问题。

## 4. 语言模型的简史

神经网络之前：

- Shannon 用语言模型来研究英语的信息熵。
- N-gram 模型曾用于机器翻译和语音识别。

神经网络阶段：

- LSTM 和早期 neural language model 开始用学习到的模型处理序列。
- Seq2seq 推动了机器翻译。
- Attention 和 Transformer 成为现代语言模型的基础。
- Adam 这类优化器让深层模型训练更可行。

Scaling 阶段：

- GPT-2 展示了流畅生成和早期 zero-shot 能力。
- Scaling laws 让模型性能随规模变化更可预测。
- GPT-3 展示了大规模下的 in-context learning。
- Chinchilla 一类工作强调模型大小和数据量之间的 compute-optimal 权衡。

开放模型阶段：

- 开放权重和开放训练项目让我们可以研究现代语言模型的做法。
- CS336 正是利用这些公开资料来讲清语言模型底层栈。

## 5. 什么是语言模型

语言模型本质上是在 token 序列上放一个概率分布。

它的使用界面在变：

- 2018：一个拿来 fine-tune 的模型。
- 2020：一个拿来 prompt 的模型。
- 2022：一个可以 chat 的模型。
- 2026：一个可以作为 agent 自主行动的模型。

但底层基础没有变：tokens、attention、optimization、kernels、data、evaluation。

## 6. 课程大纲

Assignment 1：Basics

- Tokenization
- Transformer architecture
- Cross-entropy loss
- AdamW optimizer
- Training loop
- Resource accounting

Assignment 2：Systems

- Kernels
- GPU efficiency
- Parallelism
- Inference

Assignment 3：Scaling Laws

- 预测模型大小、数据量、compute 改变时，性能会怎么变。

Assignment 4：Data

- Evaluation
- Curation
- Filtering
- Deduplication
- Data mixing

Assignment 5：Alignment

- Supervised fine-tuning
- Preference learning
- RLHF 相关算法和系统

## 7. Tokenization：第一单元

对应 notebook：第 7 节开始，先运行公共工具 cell。

Tokenization 要回答的问题是：

> 模型操作的基本单位是什么？

形式上，tokenizer 在原始文本和整数 token 序列之间转换：

- Encode：string -> token id 列表
- Decode：token id 列表 -> string

语言模型通常不直接处理 Python 字符串。它处理整数 id，然后通过 embedding table 把这些 id 映射成向量。

## 8. Character Tokenization

对应 notebook：第 8 节，运行 `CharacterTokenizer` cell。

Unicode 字符串可以看成字符序列。每个字符都有一个 code point：

- `ord("a") == 97`
- `chr(97) == "a"`

优点：

- 简单。
- 可以把文本 encode 后再 decode 回来。

缺点：

- Unicode 字符数量很大。
- 稀有字符会浪费 vocabulary 容量。
- 相比现代 tokenizer，压缩率差。

## 9. Byte Tokenization

对应 notebook：第 9 节，运行 `ByteTokenizer` cell，对比中文和 emoji 的 UTF-8 bytes。

Unicode 字符串也可以编码成 UTF-8 bytes。每个 byte 是 0 到 255 之间的整数。

优点：

- 固定小词表：256 个 byte 值。
- 可以表示任何 UTF-8 文本。

缺点：

- 压缩率差：基本是一 byte 一个 token。
- 序列会变长。
- 标准 Transformer attention 的成本随序列长度平方增长，所以长序列很贵。

## 10. Word Tokenization

对应 notebook：第 10 节，运行 toy word tokenizer，观察长词和中文的切分。

传统 NLP 常常把文本切成词或类似词的片段。

优点：

- token 对人来说有意义。
- 压缩率通常不错。

缺点：

- vocabulary 可能巨大。
- 稀有词很难学。
- 没见过的新词需要 unknown token 机制。
- 固定 vocabulary size 不自然。

## 11. Byte Pair Encoding

对应 notebook：第 13 到 15 节，依次运行 `merge`、`train_tiny_bpe`、`TinyBPETokenizer`。

Byte Pair Encoding，也就是 BPE，是 bytes 和 words 之间的折中方案。

基本思路：

1. 一开始把每个 byte 当成 token。
2. 统计训练文本里相邻 token pair 的频率。
3. 把最常见的相邻 pair 合并成一个新 token。
4. 重复，直到 vocabulary 达到目标大小。

直觉：

- 常见 byte 序列会变成单个 token。
- 罕见序列仍然拆成更小的片段。
- tokenizer 学到的是适配训练数据的 vocabulary。

这样既有固定 vocabulary，又通常比纯 byte tokenization 得到短得多的序列。

## 12. Vocabulary Size 与 Compression

对应 notebook：第 11 到 12 节，先用 `tiktoken` 观察真实 tokenizer，再比较 compression ratio。

这里有两个相互拉扯的因素：

- vocabulary 越大，常见片段越可能变成单 token，压缩率可能更好。
- vocabulary 越大，模型的 embedding 层和输出层也越大，而且更稀疏。

第一讲里的 compression ratio 指：

```text
UTF-8 byte 数量 / token 数量
```

ratio 越高，表示同样文本需要的 token 越少，通常对 attention 成本更友好。

## 13. Assignment 1 预览

Assignment 1 会让你实现：

- BPE tokenizer
- Transformer
- Cross-entropy loss
- AdamW
- Training loop
- Resource accounting

要一直平衡三件事：

- Expressivity：模型能不能表示复杂依赖。
- Stability：activation 和 gradient 是否保持在合理范围。
- Efficiency：训练和推理在真实硬件上是否高效。

## 14. 和 Week 1 的关系

对应 notebook：第 17 到 18 节，最后用检查点问题确认今天是否学完。

你 Week 1 跑的 tiny MLP 不是语言模型，但它训练的是同一个核心闭环：

```text
input -> model -> prediction -> loss -> backward -> optimizer step
```

在语言模型里，它会变成：

```text
token ids -> Transformer -> logits -> cross-entropy loss -> backward -> optimizer step
```

所以第一周不要求你掌握 tokenization。你现在要先理解：为什么 CS336 从 tokenization 开始，以及为什么后面所有内容都离不开训练循环。
