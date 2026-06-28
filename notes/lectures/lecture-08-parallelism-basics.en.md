# Lecture 08: Parallelism Basics
Source: `external/cs336-lectures/lecture_08.pdf`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-08-parallelism-basics.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲从网络和集群视角补充分布式训练：为什么多种并行方式经常组合使用，以及大规模训练 run 通常长什么样。

## 1. 本讲路线图

Notebook pointer: section 1.

网络基础、并行策略、组合并行、大规模训练实践。

## 2. 网络不是免费的

Notebook pointer: section 2.

跨 GPU/节点通信有带宽和延迟；通信模式会限制扩展效率。

## 3. 为什么组合并行

Notebook pointer: section 3.

data parallel 解决吞吐，tensor parallel 解决单层太大，pipeline/FSDP 解决显存。

## 4. FSDP / ZeRO 直觉

Notebook pointer: section 4.

把参数、梯度、optimizer state 分片，必要时 all-gather，用完释放。

## 5. 大 run 的现实

Notebook pointer: section 5.

训练不是只启动一次脚本，还包括 checkpoint、故障恢复、监控、数据吞吐和性能回归。

## Checkpoints

- 能解释为什么通信带宽会限制扩展。
- 能说出 FSDP/ZeRO 通过分片省了什么。
- 能理解大规模训练需要 checkpoint 和故障恢复。
