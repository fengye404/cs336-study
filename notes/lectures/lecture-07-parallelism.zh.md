# 第 7 讲：并行训练
来源：`external/cs336-lectures/lecture_07.py`

这是 Lecture 07 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-07-parallelism.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲讲分布式训练的通信积木和训练并行：broadcast、reduce、all-reduce、all-gather、reduce-scatter，以及 data/tensor/pipeline parallelism。

## 1. 本讲路线图

对应 notebook：第 1 节。

先理解 collective communication，再把它们组合成分布式训练策略。

## 2. 为什么需要并行

对应 notebook：第 2 节。

单卡显存、算力和训练时间都有限，大模型训练必须跨 GPU。

## 3. Collectives

对应 notebook：第 3 节。

all-reduce 合并梯度，all-gather 收集分片，reduce-scatter 合并并分发结果。

## 4. Data parallelism

对应 notebook：第 4 节。

每张卡有完整模型和不同 batch，反向后 all-reduce 梯度。

## 5. Tensor parallelism

对应 notebook：第 5 节。

把大矩阵或 attention head 切到多卡，单层内部就需要通信。

## 6. Pipeline parallelism

对应 notebook：第 6 节。

把层切到不同设备，microbatch 流水线执行，换取显存但带来 bubble。

## 检查点

- 能解释 all-reduce 在 data parallel 中做什么。
- 能区分 data/tensor/pipeline parallel。
- 能说出 pipeline bubble 为什么存在。
