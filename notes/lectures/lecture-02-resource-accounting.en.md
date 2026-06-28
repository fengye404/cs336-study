# Lecture 02: Resource Accounting
Source: `external/cs336-lectures/lecture_02.py`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-02-resource-accounting.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲把训练语言模型拆成 compute、memory、dtype、FLOPs、arithmetic intensity 和训练循环的资源账。目标不是背公式，而是养成先估算资源再运行实验的习惯。

## 1. 本讲路线图

Notebook pointer: section 1.

从 tensor 和 dtype 出发，估算 memory；从矩阵乘法出发，估算 FLOPs；再把这些账接到 forward、backward、optimizer 和训练循环。

## 2. Tensor 是所有状态的容器

Notebook pointer: section 2.

数据、参数、activation、gradient、optimizer state 都是 tensor。每个 tensor 的内存由元素个数和 dtype 决定。

## 3. 浮点格式和显存

Notebook pointer: section 3.

fp32 通常 4 bytes，fp16/bf16 通常 2 bytes。混合精度训练的核心收益之一是降低显存和带宽压力。

## 4. FLOPs 和训练时间

Notebook pointer: section 4.

常用粗略估计：训练 N 个参数、D 个 token 的 dense Transformer，大约需要 6ND FLOPs。

## 5. Arithmetic intensity

Notebook pointer: section 5.

FLOPs / bytes moved。强度高的操作更可能 compute-bound，强度低的操作更可能 memory-bound。

## 6. 训练循环的资源

Notebook pointer: section 6.

参数、梯度、optimizer state、activation 都要占内存；backward 和 optimizer step 也要消耗 compute。

## Checkpoints

- 能解释为什么 dtype 会直接改变显存。
- 能用 6ND 粗估训练 FLOPs。
- 能区分 compute-bound 和 memory-bound 的直觉。
- 能说出训练时除了参数外还有哪些东西占显存。
