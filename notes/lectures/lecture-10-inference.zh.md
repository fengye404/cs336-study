# 第 10 讲：Inference
来源：`external/cs336-lectures/lecture_10.py`

这是 Lecture 10 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-10-inference.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲讲模型真正被使用时的系统问题：prefill/decode、TTFT、latency、throughput、KV cache、GQA/MLA、quantization、speculative sampling、paged attention。

## 1. 本讲路线图

对应 notebook：第 1 节。

先理解 inference workload，再看 lossy shortcut、lossless shortcut 和动态请求调度。

## 2. Prefill 与 decode

对应 notebook：第 2 节。

prefill 并行处理 prompt，decode 逐 token 生成，decode 更容易 memory-bound。

## 3. 指标

对应 notebook：第 3 节。

TTFT 看首 token 等待，latency 看单请求速度，throughput 看批量处理能力。

## 4. KV cache

对应 notebook：第 4 节。

缓存历史 K/V 避免重复计算，但长上下文和大 batch 会让 KV cache 成为显存瓶颈。

## 5. 压缩和捷径

对应 notebook：第 5 节。

GQA/MLA、quantization、pruning/distillation 都在减少 inference 成本。

## 6. 动态 workload

对应 notebook：第 6 节。

连续 batching、paged attention 解决真实服务里的变长请求和碎片化。

## 检查点

- 能区分 prefill 和 decode。
- 能解释 KV cache 为什么会占很多显存。
- 能说出 TTFT、latency、throughput 分别适合衡量什么。
- 能理解 speculative sampling 为什么需要校验。
