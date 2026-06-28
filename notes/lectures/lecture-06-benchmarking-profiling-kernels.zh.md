# 第 6 讲：Benchmark、Profiling 与 Kernel
来源：`external/cs336-lectures/lecture_06.py`

这是 Lecture 06 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-06-benchmarking-profiling-kernels.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲从 GPU 和性能直觉进入实际优化：怎么 benchmark、profile，什么时候写 Triton kernel，以及 elementwise、softmax、row sum、matmul tiling 的模式。

## 1. 本讲路线图

对应 notebook：第 1 节。

先复习 GPU 编程模型，再学习 benchmark/profile，最后看 Triton kernel 的几类模式。

## 2. Benchmark

对应 notebook：第 2 节。

benchmark 衡量一个操作在给定输入规模下跑多快。要注意 warmup、同步和输入规模。

## 3. Profiling

对应 notebook：第 3 节。

profiling 告诉你时间花在哪里：kernel launch、memory copy、matmul、elementwise、fusion 等。

## 4. Kernel fusion

对应 notebook：第 4 节。

把多个 elementwise 操作合并，减少中间 tensor 和 HBM 往返。

## 5. Triton 思维

对应 notebook：第 5 节。

按 program/block 处理一块数据，读入 SRAM/shared memory，计算，再写回。

## 6. Tiling

对应 notebook：第 6 节。

matmul 和 softmax 这类操作需要分块，以便复用数据并控制内存流量。

## 检查点

- 能区分 benchmark 和 profiling。
- 能解释 kernel fusion 为什么能减少内存流量。
- 能说出 Triton block/program 大致对应什么。
