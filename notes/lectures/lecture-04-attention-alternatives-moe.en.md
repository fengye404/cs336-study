# Lecture 04: Attention Alternatives And Mixtures Of Experts
Source: `external/cs336-lectures/lecture_04.pdf`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-04-attention-alternatives-moe.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲围绕长上下文成本展开：标准 attention 是二次复杂度，于是出现 local attention、linear attention、state-space/recurrence 和 MoE 等方向。

## 1. 本讲路线图

Notebook pointer: section 1.

先看 attention 为什么贵，再看降低序列成本或增加参数容量的不同路线。

## 2. 标准 attention 的瓶颈

Notebook pointer: section 2.

QK^T 需要 n^2 级别的 token-token 交互，长上下文下成本迅速上升。

## 3. Local / sparse attention

Notebook pointer: section 3.

只看局部窗口或少量全局 token，用表达能力换效率。

## 4. Linear attention

Notebook pointer: section 4.

通过改写或近似 attention，把二次 token 交互变成线性或近线性成本。

## 5. MoE

Notebook pointer: section 5.

Mixture of Experts 用 router 选择少数 expert，让总参数变大但每个 token 只激活一部分。

## 6. 核心 tradeoff

Notebook pointer: section 6.

这些方法都在表达能力、训练稳定性、硬件效率和工程复杂度之间做交换。

## Checkpoints

- 能解释标准 attention 为什么是 O(n^2)。
- 能说出 local attention 省在哪里、损失在哪里。
- 能解释 MoE 的 activated parameters 和 total parameters 的区别。
- 能理解 router load balancing 为什么重要。
