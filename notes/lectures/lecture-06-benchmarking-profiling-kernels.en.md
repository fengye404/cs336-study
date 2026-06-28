# Lecture 06: Benchmarking, Profiling, And Kernels
Source: `external/cs336-lectures/lecture_06.py`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-06-benchmarking-profiling-kernels.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲从 GPU 和性能直觉进入实际优化：怎么 benchmark、profile，什么时候写 Triton kernel，以及 elementwise、softmax、row sum、matmul tiling 的模式。

## 1. 本讲路线图

Notebook pointer: section 1.

先复习 GPU 编程模型，再学习 benchmark/profile，最后看 Triton kernel 的几类模式。

## 2. Benchmark

Notebook pointer: section 2.

benchmark 衡量一个操作在给定输入规模下跑多快。要注意 warmup、同步和输入规模。

## 3. Profiling

Notebook pointer: section 3.

profiling 告诉你时间花在哪里：kernel launch、memory copy、matmul、elementwise、fusion 等。

## 4. Kernel fusion

Notebook pointer: section 4.

把多个 elementwise 操作合并，减少中间 tensor 和 HBM 往返。

## 5. Triton 思维

Notebook pointer: section 5.

按 program/block 处理一块数据，读入 SRAM/shared memory，计算，再写回。

## 6. Tiling

Notebook pointer: section 6.

matmul 和 softmax 这类操作需要分块，以便复用数据并控制内存流量。

## Checkpoints

- 能区分 benchmark 和 profiling。
- 能解释 kernel fusion 为什么能减少内存流量。
- 能说出 Triton block/program 大致对应什么。
