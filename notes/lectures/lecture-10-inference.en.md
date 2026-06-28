# Lecture 10: Inference
Source: `external/cs336-lectures/lecture_10.py`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-10-inference.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲讲模型真正被使用时的系统问题：prefill/decode、TTFT、latency、throughput、KV cache、GQA/MLA、quantization、speculative sampling、paged attention。

## 1. 本讲路线图

Notebook pointer: section 1.

先理解 inference workload，再看 lossy shortcut、lossless shortcut 和动态请求调度。

## 2. Prefill 与 decode

Notebook pointer: section 2.

prefill 并行处理 prompt，decode 逐 token 生成，decode 更容易 memory-bound。

## 3. 指标

Notebook pointer: section 3.

TTFT 看首 token 等待，latency 看单请求速度，throughput 看批量处理能力。

## 4. KV cache

Notebook pointer: section 4.

缓存历史 K/V 避免重复计算，但长上下文和大 batch 会让 KV cache 成为显存瓶颈。

## 5. 压缩和捷径

Notebook pointer: section 5.

GQA/MLA、quantization、pruning/distillation 都在减少 inference 成本。

## 6. 动态 workload

Notebook pointer: section 6.

连续 batching、paged attention 解决真实服务里的变长请求和碎片化。

## Checkpoints

- 能区分 prefill 和 decode。
- 能解释 KV cache 为什么会占很多显存。
- 能说出 TTFT、latency、throughput 分别适合衡量什么。
- 能理解 speculative sampling 为什么需要校验。
