# 第 8 讲：Parallelism Basics
来源：`external/cs336-lectures/lecture_08.pdf`

这是 Lecture 08 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-08-parallelism-basics.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲从网络和集群视角补充分布式训练：为什么多种并行方式经常组合使用，以及大规模训练 run 通常长什么样。

## 1. 本讲路线图

对应 notebook：第 1 节。

网络基础、并行策略、组合并行、大规模训练实践。

## 2. 网络不是免费的

对应 notebook：第 2 节。

跨 GPU/节点通信有带宽和延迟；通信模式会限制扩展效率。

## 3. 为什么组合并行

对应 notebook：第 3 节。

data parallel 解决吞吐，tensor parallel 解决单层太大，pipeline/FSDP 解决显存。

## 4. FSDP / ZeRO 直觉

对应 notebook：第 4 节。

把参数、梯度、optimizer state 分片，必要时 all-gather，用完释放。

## 5. 大 run 的现实

对应 notebook：第 5 节。

训练不是只启动一次脚本，还包括 checkpoint、故障恢复、监控、数据吞吐和性能回归。

## 检查点

- 能解释为什么通信带宽会限制扩展。
- 能说出 FSDP/ZeRO 通过分片省了什么。
- 能理解大规模训练需要 checkpoint 和故障恢复。
