# Lecture 05: GPUs
Source: `external/cs336-lectures/lecture_05.pdf`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-05-gpus.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲把 GPU 从黑盒拆开：SM、warp、register、shared memory、HBM、roofline，以及为什么 FlashAttention 能快。

## 1. 本讲路线图

Notebook pointer: section 1.

先理解硬件结构，再理解性能瓶颈，最后用 FlashAttention 串起来。

## 2. GPU 为什么适合深度学习

Notebook pointer: section 2.

GPU 用大量并行线程和高带宽内存服务矩阵乘法、attention、elementwise kernel。

## 3. Memory hierarchy

Notebook pointer: section 3.

register 最快最小，shared memory 局部共享，HBM 大但慢；优化常常是减少 HBM 往返。

## 4. Warp 与 occupancy

Notebook pointer: section 4.

warp 是 32 个线程的执行单位；occupancy 影响隐藏内存延迟的能力。

## 5. Roofline

Notebook pointer: section 5.

性能上限取决于 compute 峰值和 memory bandwidth，两者由 arithmetic intensity 连接。

## 6. FlashAttention 直觉

Notebook pointer: section 6.

不显式 materialize 巨大的 attention matrix，而是分块计算，减少 HBM 读写。

## Checkpoints

- 能解释 HBM、shared memory、register 的区别。
- 能用 roofline 判断一个操作更像 memory-bound 还是 compute-bound。
- 能说出 FlashAttention 为什么不是改数学结果，而是改数据移动方式。
