# Lecture 07: Parallelism
Source: `external/cs336-lectures/lecture_07.py`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-07-parallelism.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲讲分布式训练的通信积木和训练并行：broadcast、reduce、all-reduce、all-gather、reduce-scatter，以及 data/tensor/pipeline parallelism。

## 1. 本讲路线图

Notebook pointer: section 1.

先理解 collective communication，再把它们组合成分布式训练策略。

## 2. 为什么需要并行

Notebook pointer: section 2.

单卡显存、算力和训练时间都有限，大模型训练必须跨 GPU。

## 3. Collectives

Notebook pointer: section 3.

all-reduce 合并梯度，all-gather 收集分片，reduce-scatter 合并并分发结果。

## 4. Data parallelism

Notebook pointer: section 4.

每张卡有完整模型和不同 batch，反向后 all-reduce 梯度。

## 5. Tensor parallelism

Notebook pointer: section 5.

把大矩阵或 attention head 切到多卡，单层内部就需要通信。

## 6. Pipeline parallelism

Notebook pointer: section 6.

把层切到不同设备，microbatch 流水线执行，换取显存但带来 bubble。

## Checkpoints

- 能解释 all-reduce 在 data parallel 中做什么。
- 能区分 data/tensor/pipeline parallel。
- 能说出 pipeline bubble 为什么存在。
