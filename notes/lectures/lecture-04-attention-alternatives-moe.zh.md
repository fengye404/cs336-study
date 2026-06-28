# 第 4 讲：Attention 替代方案与 MoE
来源：`external/cs336-lectures/lecture_04.pdf`

这是 Lecture 04 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-04-attention-alternatives-moe.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲围绕长上下文成本展开：标准 attention 是二次复杂度，于是出现 local attention、linear attention、state-space/recurrence 和 MoE 等方向。

## 1. 本讲路线图

对应 notebook：第 1 节。

先看 attention 为什么贵，再看降低序列成本或增加参数容量的不同路线。

## 2. 标准 attention 的瓶颈

对应 notebook：第 2 节。

QK^T 需要 n^2 级别的 token-token 交互，长上下文下成本迅速上升。

## 3. Local / sparse attention

对应 notebook：第 3 节。

只看局部窗口或少量全局 token，用表达能力换效率。

## 4. Linear attention

对应 notebook：第 4 节。

通过改写或近似 attention，把二次 token 交互变成线性或近线性成本。

## 5. MoE

对应 notebook：第 5 节。

Mixture of Experts 用 router 选择少数 expert，让总参数变大但每个 token 只激活一部分。

## 6. 核心 tradeoff

对应 notebook：第 6 节。

这些方法都在表达能力、训练稳定性、硬件效率和工程复杂度之间做交换。

## 检查点

- 能解释标准 attention 为什么是 O(n^2)。
- 能说出 local attention 省在哪里、损失在哪里。
- 能解释 MoE 的 activated parameters 和 total parameters 的区别。
- 能理解 router load balancing 为什么重要。
