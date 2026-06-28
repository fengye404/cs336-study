from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notes" / "lectures"


def lines(text: str) -> list[str]:
    return textwrap.dedent(text).strip().splitlines(keepends=True) + ["\n"]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


LECTURES = [
    {
        "num": 2,
        "slug": "resource-accounting",
        "title_zh": "资源核算",
        "title_en": "Resource Accounting",
        "source": "external/cs336-lectures/lecture_02.py",
        "official_kind": "executable lecture",
        "summary_zh": "这一讲把训练语言模型拆成 compute、memory、dtype、FLOPs、arithmetic intensity 和训练循环的资源账。目标不是背公式，而是养成先估算资源再运行实验的习惯。",
        "sections": [
            ("本讲路线图", "从 tensor 和 dtype 出发，估算 memory；从矩阵乘法出发，估算 FLOPs；再把这些账接到 forward、backward、optimizer 和训练循环。"),
            ("Tensor 是所有状态的容器", "数据、参数、activation、gradient、optimizer state 都是 tensor。每个 tensor 的内存由元素个数和 dtype 决定。"),
            ("浮点格式和显存", "fp32 通常 4 bytes，fp16/bf16 通常 2 bytes。混合精度训练的核心收益之一是降低显存和带宽压力。"),
            ("FLOPs 和训练时间", "常用粗略估计：训练 N 个参数、D 个 token 的 dense Transformer，大约需要 6ND FLOPs。"),
            ("Arithmetic intensity", "FLOPs / bytes moved。强度高的操作更可能 compute-bound，强度低的操作更可能 memory-bound。"),
            ("训练循环的资源", "参数、梯度、optimizer state、activation 都要占内存；backward 和 optimizer step 也要消耗 compute。"),
        ],
        "codes": [
            ("显存账：dtype 与 tensor size", r'''
                def tensor_bytes(shape, bytes_per_value):
                    n = 1
                    for dim in shape:
                        n *= dim
                    return n * bytes_per_value

                shape = (32, 2048, 4096)
                for dtype, b in [("fp32", 4), ("fp16/bf16", 2), ("fp8", 1)]:
                    gb = tensor_bytes(shape, b) / 1e9
                    print(dtype, f"{gb:.2f} GB")
            '''),
            ("训练 FLOPs 粗算", r'''
                def training_flops(num_params, num_tokens):
                    return 6 * num_params * num_tokens

                params = 7e9
                tokens = 1e12
                flops = training_flops(params, tokens)
                h100_bf16_flops = 989e12  # rough effective order, not a promise
                gpus = 128
                mfu = 0.4
                seconds = flops / (h100_bf16_flops * gpus * mfu)
                print("total FLOPs:", f"{flops:.2e}")
                print("rough days:", seconds / 86400)
            '''),
            ("Arithmetic intensity 玩具例子", r'''
                def intensity(flops, bytes_moved):
                    return flops / bytes_moved

                ops = {
                    "elementwise add": (1_000_000, 12_000_000),
                    "matmul-ish": (2_000_000_000, 32_000_000),
                }
                for name, (flops, bytes_moved) in ops.items():
                    print(name, "intensity=", intensity(flops, bytes_moved))
            '''),
        ],
        "checkpoints": [
            "能解释为什么 dtype 会直接改变显存。",
            "能用 6ND 粗估训练 FLOPs。",
            "能区分 compute-bound 和 memory-bound 的直觉。",
            "能说出训练时除了参数外还有哪些东西占显存。",
        ],
    },
    {
        "num": 3,
        "slug": "architecture-hyperparameters",
        "title_zh": "语言模型架构与超参数",
        "title_en": "LM Architecture And Hyperparameters",
        "source": "external/cs336-lectures/lecture_03.pdf",
        "official_kind": "PDF lecture",
        "summary_zh": "这一讲从现代 Transformer 的常见组件出发，讨论大模型架构里哪些选择已经变成默认项，哪些仍然依赖实验经验。",
        "sections": [
            ("本讲路线图", "复习现代 Transformer，再看 RoPE、pre-norm、SwiGLU、bias、normalization、宽深比例和训练超参数。"),
            ("原始 Transformer 到现代变体", "现代 decoder-only LM 通常使用 causal self-attention、pre-norm、RoPE、SwiGLU 和无 bias 的线性层。"),
            ("Normalization", "LayerNorm 和 RMSNorm 都用于稳定 activation 尺度；RMSNorm 更简单，少了均值中心化。"),
            ("位置编码", "RoPE 把位置信息注入 Q/K 的旋转结构中，让相对位置信息影响 attention 分数。"),
            ("FFN 与 SwiGLU", "现代 LM 常用 gated FFN，例如 SwiGLU，用门控控制信息流。"),
            ("超参数不是装饰", "depth、width、heads、learning rate、batch size、初始化都会影响稳定性、效率和最终 loss。"),
        ],
        "codes": [
            ("RMSNorm 与 LayerNorm", r'''
                import numpy as np

                x = np.array([1.0, 2.0, 10.0])
                layer_norm = (x - x.mean()) / np.sqrt(((x - x.mean()) ** 2).mean() + 1e-5)
                rms_norm = x / np.sqrt((x ** 2).mean() + 1e-5)
                print("x:        ", x)
                print("LayerNorm:", layer_norm)
                print("RMSNorm:  ", rms_norm)
            '''),
            ("SwiGLU 的门控直觉", r'''
                import math

                def sigmoid(x):
                    return 1 / (1 + math.exp(-x))

                def silu(x):
                    return x * sigmoid(x)

                xs = [-3, -1, 0, 1, 3]
                for x in xs:
                    gate = silu(x)
                    value = x
                    print(x, "gate=", round(gate, 3), "gated value=", round(gate * value, 3))
            '''),
            ("RoPE 的二维旋转玩具版", r'''
                import math
                import numpy as np

                def rotate2d(v, position, theta=0.2):
                    angle = position * theta
                    R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                    return R @ np.array(v)

                q = [1.0, 0.0]
                for pos in range(5):
                    print(pos, rotate2d(q, pos))
            '''),
        ],
        "checkpoints": [
            "能画出 decoder-only Transformer block 的主要路径。",
            "能解释 pre-norm 为什么和训练稳定性有关。",
            "能说出 RoPE 在什么位置作用。",
            "能区分 architecture choice 和 hyperparameter choice。",
        ],
    },
    {
        "num": 4,
        "slug": "attention-alternatives-moe",
        "title_zh": "Attention 替代方案与 MoE",
        "title_en": "Attention Alternatives And Mixtures Of Experts",
        "source": "external/cs336-lectures/lecture_04.pdf",
        "official_kind": "PDF lecture",
        "summary_zh": "这一讲围绕长上下文成本展开：标准 attention 是二次复杂度，于是出现 local attention、linear attention、state-space/recurrence 和 MoE 等方向。",
        "sections": [
            ("本讲路线图", "先看 attention 为什么贵，再看降低序列成本或增加参数容量的不同路线。"),
            ("标准 attention 的瓶颈", "QK^T 需要 n^2 级别的 token-token 交互，长上下文下成本迅速上升。"),
            ("Local / sparse attention", "只看局部窗口或少量全局 token，用表达能力换效率。"),
            ("Linear attention", "通过改写或近似 attention，把二次 token 交互变成线性或近线性成本。"),
            ("MoE", "Mixture of Experts 用 router 选择少数 expert，让总参数变大但每个 token 只激活一部分。"),
            ("核心 tradeoff", "这些方法都在表达能力、训练稳定性、硬件效率和工程复杂度之间做交换。"),
        ],
        "codes": [
            ("Attention 成本随序列长度增长", r'''
                for n in [1024, 4096, 16384, 65536]:
                    full = n * n
                    local = n * 512
                    print(f"n={n:6d} full={full/1e6:10.1f}M local_window={local/1e6:8.1f}M")
            '''),
            ("Top-1 MoE router 玩具版", r'''
                import random

                random.seed(0)
                num_tokens = 20
                num_experts = 4
                assignments = [random.randrange(num_experts) for _ in range(num_tokens)]
                load = {e: assignments.count(e) for e in range(num_experts)}
                print("assignments:", assignments)
                print("expert load:", load)
            '''),
            ("Load balancing 损失的直觉", r'''
                loads = [8, 7, 4, 1]
                ideal = sum(loads) / len(loads)
                imbalance = sum((x - ideal) ** 2 for x in loads) / len(loads)
                print("loads:", loads)
                print("ideal:", ideal)
                print("imbalance penalty:", imbalance)
            '''),
        ],
        "checkpoints": [
            "能解释标准 attention 为什么是 O(n^2)。",
            "能说出 local attention 省在哪里、损失在哪里。",
            "能解释 MoE 的 activated parameters 和 total parameters 的区别。",
            "能理解 router load balancing 为什么重要。",
        ],
    },
    {
        "num": 5,
        "slug": "gpus",
        "title_zh": "GPU 基础",
        "title_en": "GPUs",
        "source": "external/cs336-lectures/lecture_05.pdf",
        "official_kind": "PDF lecture",
        "summary_zh": "这一讲把 GPU 从黑盒拆开：SM、warp、register、shared memory、HBM、roofline，以及为什么 FlashAttention 能快。",
        "sections": [
            ("本讲路线图", "先理解硬件结构，再理解性能瓶颈，最后用 FlashAttention 串起来。"),
            ("GPU 为什么适合深度学习", "GPU 用大量并行线程和高带宽内存服务矩阵乘法、attention、elementwise kernel。"),
            ("Memory hierarchy", "register 最快最小，shared memory 局部共享，HBM 大但慢；优化常常是减少 HBM 往返。"),
            ("Warp 与 occupancy", "warp 是 32 个线程的执行单位；occupancy 影响隐藏内存延迟的能力。"),
            ("Roofline", "性能上限取决于 compute 峰值和 memory bandwidth，两者由 arithmetic intensity 连接。"),
            ("FlashAttention 直觉", "不显式 materialize 巨大的 attention matrix，而是分块计算，减少 HBM 读写。"),
        ],
        "codes": [
            ("Roofline 玩具模型", r'''
                peak_flops = 1_000
                bandwidth = 100
                for intensity in [0.1, 1, 5, 20]:
                    attainable = min(peak_flops, bandwidth * intensity)
                    bound = "memory" if bandwidth * intensity < peak_flops else "compute"
                    print("intensity", intensity, "attainable", attainable, "bound", bound)
            '''),
            ("Occupancy 粗算", r'''
                max_registers_per_sm = 65536
                max_warps_per_sm = 64
                threads_per_block = 128
                registers_per_thread = 160
                registers_per_block = threads_per_block * registers_per_thread
                blocks = max_registers_per_sm // registers_per_block
                warps = blocks * threads_per_block / 32
                print("blocks per SM:", blocks)
                print("warps per SM:", warps)
                print("occupancy:", warps / max_warps_per_sm)
            '''),
            ("Attention matrix 的内存压力", r'''
                for seq in [2048, 8192, 32768]:
                    bytes_fp16 = seq * seq * 2
                    print(seq, "attention matrix GB:", bytes_fp16 / 1e9)
            '''),
        ],
        "checkpoints": [
            "能解释 HBM、shared memory、register 的区别。",
            "能用 roofline 判断一个操作更像 memory-bound 还是 compute-bound。",
            "能说出 FlashAttention 为什么不是改数学结果，而是改数据移动方式。",
        ],
    },
    {
        "num": 6,
        "slug": "benchmarking-profiling-kernels",
        "title_zh": "Benchmark、Profiling 与 Kernel",
        "title_en": "Benchmarking, Profiling, And Kernels",
        "source": "external/cs336-lectures/lecture_06.py",
        "official_kind": "executable lecture",
        "summary_zh": "这一讲从 GPU 和性能直觉进入实际优化：怎么 benchmark、profile，什么时候写 Triton kernel，以及 elementwise、softmax、row sum、matmul tiling 的模式。",
        "sections": [
            ("本讲路线图", "先复习 GPU 编程模型，再学习 benchmark/profile，最后看 Triton kernel 的几类模式。"),
            ("Benchmark", "benchmark 衡量一个操作在给定输入规模下跑多快。要注意 warmup、同步和输入规模。"),
            ("Profiling", "profiling 告诉你时间花在哪里：kernel launch、memory copy、matmul、elementwise、fusion 等。"),
            ("Kernel fusion", "把多个 elementwise 操作合并，减少中间 tensor 和 HBM 往返。"),
            ("Triton 思维", "按 program/block 处理一块数据，读入 SRAM/shared memory，计算，再写回。"),
            ("Tiling", "matmul 和 softmax 这类操作需要分块，以便复用数据并控制内存流量。"),
        ],
        "codes": [
            ("Python 层 benchmark 的基本形状", r'''
                import math
                import timeit

                def gelu_scalar(x):
                    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

                xs = [i / 1000 for i in range(10000)]
                t = timeit.timeit(lambda: [gelu_scalar(x) for x in xs], number=20)
                print("seconds:", t)
            '''),
            ("Fusion 的内存直觉", r'''
                n = 1_000_000
                bytes_per_value = 4
                unfused_reads_writes = 6 * n * bytes_per_value
                fused_reads_writes = 2 * n * bytes_per_value
                print("unfused MB:", unfused_reads_writes / 1e6)
                print("fused MB:", fused_reads_writes / 1e6)
            '''),
            ("Tiling matmul 的块数", r'''
                import math

                M = N = K = 4096
                block = 128
                tiles_mn = math.ceil(M / block) * math.ceil(N / block)
                tiles_k = math.ceil(K / block)
                print("output tiles:", tiles_mn)
                print("K tiles per output tile:", tiles_k)
            '''),
        ],
        "checkpoints": [
            "能区分 benchmark 和 profiling。",
            "能解释 kernel fusion 为什么能减少内存流量。",
            "能说出 Triton block/program 大致对应什么。",
        ],
    },
    {
        "num": 7,
        "slug": "parallelism",
        "title_zh": "并行训练",
        "title_en": "Parallelism",
        "source": "external/cs336-lectures/lecture_07.py",
        "official_kind": "executable lecture",
        "summary_zh": "这一讲讲分布式训练的通信积木和训练并行：broadcast、reduce、all-reduce、all-gather、reduce-scatter，以及 data/tensor/pipeline parallelism。",
        "sections": [
            ("本讲路线图", "先理解 collective communication，再把它们组合成分布式训练策略。"),
            ("为什么需要并行", "单卡显存、算力和训练时间都有限，大模型训练必须跨 GPU。"),
            ("Collectives", "all-reduce 合并梯度，all-gather 收集分片，reduce-scatter 合并并分发结果。"),
            ("Data parallelism", "每张卡有完整模型和不同 batch，反向后 all-reduce 梯度。"),
            ("Tensor parallelism", "把大矩阵或 attention head 切到多卡，单层内部就需要通信。"),
            ("Pipeline parallelism", "把层切到不同设备，microbatch 流水线执行，换取显存但带来 bubble。"),
        ],
        "codes": [
            ("All-reduce 梯度玩具模拟", r'''
                grads = [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                    [0.5, 1.5, 2.5],
                ]
                avg = [sum(values) / len(values) for values in zip(*grads)]
                print("local grads:", grads)
                print("all-reduced average:", avg)
            '''),
            ("All-gather 与 reduce-scatter 直觉", r'''
                shards = [["p0"], ["p1"], ["p2"], ["p3"]]
                gathered = sum(shards, [])
                print("all-gather:", gathered)

                values = [10, 20, 30, 40]
                world_size = 4
                reduced_then_scattered = [sum(values) / world_size]
                print("reduce-scatter shard example:", reduced_then_scattered)
            '''),
            ("Pipeline bubble 粗略估计", r'''
                def bubble_fraction(num_stages, microbatches):
                    return (num_stages - 1) / (microbatches + num_stages - 1)

                for microbatches in [1, 2, 4, 8, 16]:
                    print(microbatches, bubble_fraction(4, microbatches))
            '''),
        ],
        "checkpoints": [
            "能解释 all-reduce 在 data parallel 中做什么。",
            "能区分 data/tensor/pipeline parallel。",
            "能说出 pipeline bubble 为什么存在。",
        ],
    },
    {
        "num": 8,
        "slug": "parallelism-basics",
        "title_zh": "Parallelism Basics",
        "title_en": "Parallelism Basics",
        "source": "external/cs336-lectures/lecture_08.pdf",
        "official_kind": "PDF lecture",
        "summary_zh": "这一讲从网络和集群视角补充分布式训练：为什么多种并行方式经常组合使用，以及大规模训练 run 通常长什么样。",
        "sections": [
            ("本讲路线图", "网络基础、并行策略、组合并行、大规模训练实践。"),
            ("网络不是免费的", "跨 GPU/节点通信有带宽和延迟；通信模式会限制扩展效率。"),
            ("为什么组合并行", "data parallel 解决吞吐，tensor parallel 解决单层太大，pipeline/FSDP 解决显存。"),
            ("FSDP / ZeRO 直觉", "把参数、梯度、optimizer state 分片，必要时 all-gather，用完释放。"),
            ("大 run 的现实", "训练不是只启动一次脚本，还包括 checkpoint、故障恢复、监控、数据吞吐和性能回归。"),
        ],
        "codes": [
            ("ZeRO 分片显存粗算", r'''
                params = 70e9
                bytes_per_param_state = 2 + 2 + 4 + 4  # param, grad, Adam moments
                for world in [1, 8, 64, 512]:
                    gb = params * bytes_per_param_state / world / 1e9
                    print(f"world={world:3d} per-rank state={gb:.1f} GB")
            '''),
            ("通信时间粗算", r'''
                def transfer_time_gb(size_gb, bandwidth_gb_s, latency_ms=0.02):
                    return latency_ms / 1000 + size_gb / bandwidth_gb_s

                for size in [0.1, 1, 10]:
                    print(size, "GB over 400GB/s:", transfer_time_gb(size, 400), "s")
            '''),
        ],
        "checkpoints": [
            "能解释为什么通信带宽会限制扩展。",
            "能说出 FSDP/ZeRO 通过分片省了什么。",
            "能理解大规模训练需要 checkpoint 和故障恢复。",
        ],
    },
    {
        "num": 9,
        "slug": "scaling-laws-basics",
        "title_zh": "Scaling Laws 基础",
        "title_en": "Scaling Laws Basics",
        "source": "external/cs336-lectures/lecture_09.pdf",
        "official_kind": "PDF lecture",
        "summary_zh": "这一讲讲如何认真对待 scaling：用小实验拟合简单预测规律，再外推到大模型，帮助选择模型大小、数据量和 compute。",
        "sections": [
            ("本讲路线图", "为什么需要 scaling laws，如何拟合，如何用它们做预算决策。"),
            ("问题设定", "给定固定训练预算，应该训练多大的模型、用多少数据、跑多久？"),
            ("Power law 直觉", "很多 loss 随 compute、参数、数据变化呈近似幂律关系，在 log-log 坐标里接近直线。"),
            ("Chinchilla 风格权衡", "模型太大数据太少会 undertrain，模型太小数据太多也浪费。"),
            ("外推的风险", "小规模规律不一定完全延伸到大规模，数据质量、架构、优化都会改变曲线。"),
        ],
        "codes": [
            ("Log-log 拟合 power law", r'''
                import numpy as np

                compute = np.array([1, 3, 10, 30, 100], dtype=float)
                loss = 2.0 * compute ** -0.08 + 1.2
                x = np.log(compute)
                y = np.log(loss - 1.2)
                slope, intercept = np.polyfit(x, y, 1)
                print("fitted exponent:", slope)
                print("predicted loss at compute=1000:", np.exp(intercept) * 1000 ** slope + 1.2)
            '''),
            ("模型大小与数据量的预算玩具例子", r'''
                budget = 1e20
                candidates = [(1e9, 1e11), (7e9, 2e11), (70e9, 2e10)]
                for params, tokens in candidates:
                    flops = 6 * params * tokens
                    print(f"params={params:.0e} tokens={tokens:.0e} fits={flops <= budget}")
            '''),
        ],
        "checkpoints": [
            "能解释为什么 log-log 图上直线代表 power law。",
            "能用 scaling law 的思路比较两个训练方案。",
            "能说出外推为什么有风险。",
        ],
    },
    {
        "num": 10,
        "slug": "inference",
        "title_zh": "Inference",
        "title_en": "Inference",
        "source": "external/cs336-lectures/lecture_10.py",
        "official_kind": "executable lecture",
        "summary_zh": "这一讲讲模型真正被使用时的系统问题：prefill/decode、TTFT、latency、throughput、KV cache、GQA/MLA、quantization、speculative sampling、paged attention。",
        "sections": [
            ("本讲路线图", "先理解 inference workload，再看 lossy shortcut、lossless shortcut 和动态请求调度。"),
            ("Prefill 与 decode", "prefill 并行处理 prompt，decode 逐 token 生成，decode 更容易 memory-bound。"),
            ("指标", "TTFT 看首 token 等待，latency 看单请求速度，throughput 看批量处理能力。"),
            ("KV cache", "缓存历史 K/V 避免重复计算，但长上下文和大 batch 会让 KV cache 成为显存瓶颈。"),
            ("压缩和捷径", "GQA/MLA、quantization、pruning/distillation 都在减少 inference 成本。"),
            ("动态 workload", "连续 batching、paged attention 解决真实服务里的变长请求和碎片化。"),
        ],
        "codes": [
            ("KV cache 大小估算", r'''
                def kv_cache_gb(layers, seq, batch, kv_heads, head_dim, bytes_per_value=2):
                    # K and V
                    return layers * seq * batch * kv_heads * head_dim * 2 * bytes_per_value / 1e9

                for seq in [4096, 32768, 131072]:
                    print(seq, "tokens:", kv_cache_gb(32, seq, 8, 8, 128), "GB")
            '''),
            ("TTFT、latency、throughput 的区别", r'''
                requests = [
                    {"prefill_s": 0.8, "decode_tokens": 100, "decode_s": 5.0},
                    {"prefill_s": 0.2, "decode_tokens": 20, "decode_s": 1.0},
                ]
                for r in requests:
                    print("TTFT", r["prefill_s"], "s")
                    print("latency/token", r["decode_s"] / r["decode_tokens"], "s/token")
            '''),
            ("Speculative sampling 接受率玩具模型", r'''
                draft_tokens = 1000
                accept_rate = 0.7
                target_calls_without = draft_tokens
                target_calls_with = draft_tokens * (1 - accept_rate)
                print("target calls without speculation:", target_calls_without)
                print("extra target work after accepted drafts:", target_calls_with)
            '''),
        ],
        "checkpoints": [
            "能区分 prefill 和 decode。",
            "能解释 KV cache 为什么会占很多显存。",
            "能说出 TTFT、latency、throughput 分别适合衡量什么。",
            "能理解 speculative sampling 为什么需要校验。",
        ],
    },
    {
        "num": 11,
        "slug": "scaling-details",
        "title_zh": "Scaling 实践与细节",
        "title_en": "Scaling Case Study And Details",
        "source": "external/cs336-lectures/lecture_11.pdf",
        "official_kind": "PDF lecture",
        "summary_zh": "这一讲继续 scaling：实际调参、初始化、optimizer、学习率、batch、参数化方式，以及如何节省拟合 scaling curve 的 compute。",
        "sections": [
            ("本讲路线图", "从 scaling law 到实践：hparam tuning、初始化、optimizer、参数化和跨尺度稳定性。"),
            ("Best practice 问题", "大模型不能盲目调参，只能用小规模实验指导大规模设置。"),
            ("初始化与参数化", "不同参数化方式会改变 activation、gradient 和 learning rate 对尺度的敏感度。"),
            ("Optimizer 与 learning rate", "AdamW、Muon、batch size、warmup、schedule 都可能随规模变化。"),
            ("节省 compute", "用小模型/短训练拟合趋势，用代理指标筛选，再把少数设置放大验证。"),
        ],
        "codes": [
            ("学习率尺度敏感性的玩具例子", r'''
                def stable_update(param_scale, lr):
                    update = lr * param_scale
                    return update < 0.1

                for scale in [1, 10, 100]:
                    for lr in [1e-3, 1e-2]:
                        print("scale", scale, "lr", lr, "stable?", stable_update(scale, lr))
            '''),
            ("小实验筛选候选超参数", r'''
                candidates = [
                    {"lr": 1e-3, "small_loss": 2.1},
                    {"lr": 3e-4, "small_loss": 1.9},
                    {"lr": 1e-4, "small_loss": 2.0},
                ]
                best = min(candidates, key=lambda x: x["small_loss"])
                print("promote to larger run:", best)
            '''),
        ],
        "checkpoints": [
            "能解释为什么大模型超参数不能只靠试错。",
            "能说出初始化/参数化为什么影响 scaling。",
            "能描述一个小实验到大实验的筛选流程。",
        ],
    },
    {
        "num": 12,
        "slug": "evaluation",
        "title_zh": "Evaluation",
        "title_en": "Evaluation",
        "source": "external/cs336-lectures/lecture_12.py",
        "official_kind": "executable lecture",
        "summary_zh": "这一讲讨论 evaluation 的目标、对象、数据质量和污染问题。评估不是跑几个 benchmark，而是定义你真正关心的能力和失效模式。",
        "sections": [
            ("本讲路线图", "评估什么、怎么评估、数据质量、train-test overlap、benchmark 的局限。"),
            ("为什么 evaluation 难", "模型能力是多维的；benchmark 容易被污染、过拟合或测不到真实使用场景。"),
            ("评估对象", "可能评估 base LM、chat model、agent、tool-use system，也可能评估 latency/cost/safety。"),
            ("Train-test overlap", "训练数据如果包含测试题，分数会虚高，无法反映泛化。"),
            ("数据质量", "题目是否清楚、标注是否一致、答案是否唯一，都会影响评估信号。"),
            ("评估的用途", "选择模型、发现失效模式、监控回归、指导数据和训练改进。"),
        ],
        "codes": [
            ("Exact match 与归一化", r'''
                import re

                def normalize(s):
                    return re.sub(r"\\s+", " ", s.strip().lower())

                gold = ["Paris", "42", "hello world"]
                pred = [" paris ", "forty two", "Hello   World"]
                scores = [normalize(g) == normalize(p) for g, p in zip(gold, pred)]
                print(scores)
                print("accuracy:", sum(scores) / len(scores))
            '''),
            ("Train-test overlap 玩具检测", r'''
                train = {"what is 2+2?", "capital of france?", "write a hello world program"}
                test = {"capital of france?", "who wrote hamlet?"}
                print("overlap:", train & test)
            '''),
            ("聚合指标会隐藏细分失败", r'''
                by_category = {"math": [1, 0, 0], "code": [1, 1, 1], "history": [0, 1]}
                for cat, scores in by_category.items():
                    print(cat, sum(scores) / len(scores))
                all_scores = sum(by_category.values(), [])
                print("overall", sum(all_scores) / len(all_scores))
            '''),
        ],
        "checkpoints": [
            "能解释 benchmark 分数为什么不是模型质量本身。",
            "能说出 train-test overlap 的危害。",
            "能设计一个小的 agent 场景 eval。",
        ],
    },
    {
        "num": 13,
        "slug": "data-i",
        "title_zh": "Data I",
        "title_en": "Data I",
        "source": "external/cs336-lectures/lecture_13.py",
        "official_kind": "executable lecture",
        "summary_zh": "这一讲进入数据：数据来源、法律与许可、爬取、过滤、DCLM/CommonCrawl 这类数据 pipeline 的目标和风险。",
        "sections": [
            ("本讲路线图", "从数据来源和法律问题开始，到自动构建大规模高质量预训练数据集。"),
            ("数据为什么重要", "pretraining loss 和能力很大程度取决于 token 来自哪里、质量如何、比例如何。"),
            ("法律与许可", "公开可访问不等于可自由使用；license、robots、copyright 都影响数据策略。"),
            ("Crawling 与 extraction", "网页需要抓取、抽正文、去模板、去广告、去噪声。"),
            ("Filtering", "规则过滤、质量模型过滤、语言过滤、安全过滤会改变数据分布。"),
            ("数据集是系统", "数据 pipeline 要可复现、可审计、可迭代，而不是一次性脚本。"),
        ],
        "codes": [
            ("简单质量过滤", r'''
                docs = [
                    "This is a clear paragraph about transformers.",
                    "BUY BUY BUY !!! $$$",
                    "def hello(): print('hi')",
                    "the the the the the",
                ]

                def quality_score(text):
                    alpha = sum(ch.isalpha() for ch in text)
                    unique = len(set(text.split()))
                    return alpha / max(len(text), 1) + unique / 20

                for d in docs:
                    print(round(quality_score(d), 3), repr(d))
            '''),
            ("URL/license metadata 的重要性", r'''
                records = [
                    {"url": "https://example.com/a", "license": "cc-by", "text": "ok"},
                    {"url": "https://example.com/b", "license": "unknown", "text": "maybe not"},
                ]
                usable = [r for r in records if r["license"] in {"cc-by", "public-domain"}]
                print(usable)
            '''),
        ],
        "checkpoints": [
            "能解释为什么数据质量不是单一分数。",
            "能说出过滤会改变数据分布。",
            "能理解 license 和来源记录为什么重要。",
        ],
    },
    {
        "num": 14,
        "slug": "data-ii",
        "title_zh": "Data II",
        "title_en": "Data II",
        "source": "external/cs336-lectures/lecture_14.py",
        "official_kind": "executable lecture",
        "summary_zh": "这一讲继续数据处理，重点是语言识别、数学数据构建、去重、Jaccard similarity、MinHash 和 LSH。",
        "sections": [
            ("本讲路线图", "从找特定语言/领域文本，到用相似度和哈希做大规模去重。"),
            ("语言与领域过滤", "训练数据通常要筛出特定语言、数学、代码或高质量网页。"),
            ("重复数据的危害", "重复会浪费 compute、改变分布，也可能加剧 benchmark contamination。"),
            ("Jaccard similarity", "用集合交并比衡量文档相似度，常用于 shingles。"),
            ("MinHash", "用多个随机哈希近似 Jaccard，避免两两比较所有文档。"),
            ("LSH", "把相似文档放到同桶里，只比较候选近邻。"),
        ],
        "codes": [
            ("Jaccard similarity", r'''
                def shingles(text, k=3):
                    toks = text.lower().split()
                    return set(tuple(toks[i:i+k]) for i in range(max(0, len(toks)-k+1)))

                def jaccard(a, b):
                    return len(a & b) / len(a | b) if a | b else 1.0

                a = shingles("the quick brown fox jumps over the lazy dog")
                b = shingles("the quick brown fox jumps over a lazy dog")
                print(jaccard(a, b))
            '''),
            ("MinHash 玩具实现", r'''
                import random

                def minhash(items, salts):
                    sig = []
                    for salt in salts:
                        sig.append(min(hash((salt, item)) for item in items))
                    return sig

                salts = list(range(20))
                A = {"a", "b", "c", "d"}
                B = {"a", "b", "c", "x"}
                sigA = minhash(A, salts)
                sigB = minhash(B, salts)
                estimate = sum(x == y for x, y in zip(sigA, sigB)) / len(salts)
                print("estimated Jaccard:", estimate)
                print("true Jaccard:", len(A & B) / len(A | B))
            '''),
        ],
        "checkpoints": [
            "能手算 Jaccard similarity。",
            "能解释 MinHash 为什么能近似 Jaccard。",
            "能说出去重和 contamination 的关系。",
        ],
    },
    {
        "num": 15,
        "slug": "post-training-i",
        "title_zh": "After Pretraining / Post-training I",
        "title_en": "After Pretraining",
        "source": "external/cs336-lectures/lecture_15.pdf",
        "official_kind": "PDF lecture",
        "summary_zh": "这一讲讲预训练之后怎么让模型更可控：instruction data、SFT、偏好数据、RLHF/DPO 等 post-training 方法。",
        "sections": [
            ("本讲路线图", "从 GPT-3 式 base model 到 InstructGPT 式可控模型，需要收集目标行为数据并优化模型。"),
            ("为什么预训练不够", "预训练学到 next-token prediction，但用户需要遵循指令、拒绝不当请求、稳定格式和有用回答。"),
            ("Instruction data", "SFT 数据通常是 prompt-response 对，直接教模型模仿目标行为。"),
            ("Preference data", "偏好数据是同一 prompt 下多个回答的相对好坏，用于 reward model、DPO 或 RLHF。"),
            ("SFT 与偏好优化", "SFT 学会格式和基础行为；偏好优化进一步推动更符合人类或规则偏好的输出。"),
            ("规模问题", "post-training 数据更贵、更少，但对模型交互行为影响很大。"),
        ],
        "codes": [
            ("SFT 样本格式", r'''
                example = {
                    "messages": [
                        {"role": "user", "content": "Explain tokenization simply."},
                        {"role": "assistant", "content": "Tokenization turns text into integer tokens that a model can process."},
                    ]
                }
                print(example)
            '''),
            ("DPO loss 形状玩具版", r'''
                import math

                def log_sigmoid(x):
                    return -math.log1p(math.exp(-x))

                beta = 0.1
                chosen_delta = 3.0 - 2.0
                rejected_delta = 1.0 - 1.5
                dpo_loss = -log_sigmoid(beta * (chosen_delta - rejected_delta))
                print("DPO loss:", dpo_loss)
            '''),
        ],
        "checkpoints": [
            "能区分 pretraining、SFT、preference optimization。",
            "能描述一条 instruction tuning 数据长什么样。",
            "能解释 preference pair 比单个回答多了什么监督信号。",
        ],
    },
    {
        "num": 16,
        "slug": "post-training-ii-rlvr",
        "title_zh": "Post-training II：可验证奖励 RL",
        "title_en": "Post-training II: RL From Verifiable Rewards",
        "source": "external/cs336-lectures/lecture_16.pdf",
        "official_kind": "PDF lecture",
        "summary_zh": "这一讲从 RLHF 走到可验证奖励：在数学、代码等领域可以用明确 reward 训练长推理行为，涉及 PPO、GRPO 和 long-CoT 现象。",
        "sections": [
            ("本讲路线图", "核心算法、案例研究、现象：PPO 到 GRPO，可验证奖励，长 CoT，SFT vs RL。"),
            ("为什么可验证奖励重要", "如果答案能自动验证，就不完全依赖人类偏好模型，RL 信号更清楚也更容易扩展。"),
            ("PPO / GRPO 直觉", "策略生成多个回答，根据 reward 调整概率，同时限制模型不要偏离太远。"),
            ("Long-CoT", "RL 可能鼓励模型生成更长的推理轨迹，只要这些轨迹提高可验证任务得分。"),
            ("SFT vs RL", "SFT 模仿已有答案；RL 可以探索新策略，但更不稳定。"),
            ("系统问题", "RL 需要大量采样、打分和训练循环，inference 成本也成为训练成本的一部分。"),
        ],
        "codes": [
            ("Group-relative advantage 玩具版", r'''
                rewards = [0, 1, 1, 0, 2]
                mean = sum(rewards) / len(rewards)
                advantages = [r - mean for r in rewards]
                print("rewards:", rewards)
                print("mean:", mean)
                print("advantages:", advantages)
            '''),
            ("带 KL 惩罚的 reward", r'''
                task_reward = 1.0
                kl_to_reference = 0.3
                beta = 0.05
                objective = task_reward - beta * kl_to_reference
                print("objective:", objective)
            '''),
        ],
        "checkpoints": [
            "能解释 verifiable reward 和 human preference 的区别。",
            "能说出 GRPO 的 group-relative 信号直觉。",
            "能理解为什么 RL 训练会增加 inference 成本。",
        ],
    },
    {
        "num": 17,
        "slug": "multimodal-models",
        "title_zh": "多模态模型",
        "title_en": "Multimodal Models",
        "source": "external/cs336-lectures/lecture_17.py",
        "official_kind": "executable lecture",
        "summary_zh": "这一讲讲语言模型如何接入图像、音频、视频等模态：把不同模态转成 token-like 表示，再通过 projector、encoder 或统一架构和语言模型连接。",
        "sections": [
            ("本讲路线图", "多模态输入、视觉 encoder、patch token、CLIP/VLM、统一 token 序列和训练阶段。"),
            ("核心问题", "语言模型吃 token 序列，图像/音频/视频必须先变成可对齐的向量序列。"),
            ("Vision Transformer", "把图像切成 patch，每个 patch 类似一个视觉 token，再加位置编码进入 Transformer。"),
            ("CLIP 直觉", "用图文对比学习把图像和文本映射到同一个语义空间。"),
            ("VLM 连接方式", "常见方法是视觉 encoder + projector + LLM，或者更统一的 multimodal token 模型。"),
            ("Token budget", "图像和视频会消耗大量 token/patch，压缩和对齐很重要。"),
        ],
        "codes": [
            ("图像 patchify 玩具版", r'''
                import numpy as np

                image = np.arange(4 * 4).reshape(4, 4)
                patch_size = 2
                patches = []
                for i in range(0, image.shape[0], patch_size):
                    for j in range(0, image.shape[1], patch_size):
                        patches.append(image[i:i+patch_size, j:j+patch_size].reshape(-1))
                print(image)
                print("patches:")
                for p in patches:
                    print(p)
            '''),
            ("视频 token budget 粗算", r'''
                frames = 32
                patches_per_frame = 14 * 14
                visual_tokens = frames * patches_per_frame
                text_tokens = 512
                print("visual tokens:", visual_tokens)
                print("total tokens:", visual_tokens + text_tokens)
            '''),
        ],
        "checkpoints": [
            "能解释为什么图像要 patchify。",
            "能说出视觉 encoder + projector + LLM 的基本结构。",
            "能理解多模态模型为什么容易受 token budget 限制。",
        ],
    },
]


def build_notebook(lecture: dict) -> dict:
    n = lecture["num"]
    title = f"第 {n} 讲：{lecture['title_zh']}"
    nb_cells: list[dict] = [
        md_cell(
            f"""
            # {title}

            这是 CS336 Lecture {n:02d} 的可执行中文学习版。

            - 官方来源：`{lecture['source']}`
            - 官方形式：{lecture['official_kind']}
            - 英文标题：{lecture['title_en']}

            这份 notebook 是学习版，不是逐字翻译。它按官方讲义的结构和节奏整理主线，并在适合的位置加入小实验。
            """
        ),
        md_cell(
            f"""
            ## 0. 本讲你要抓住什么

            {lecture['summary_zh']}

            学习方式：

            1. 先读每节的中文解释。
            2. 运行对应代码 cell。
            3. 改一个参数，观察输出如何变化。
            4. 最后用检查点问题自测。
            """
        ),
    ]

    for idx, (heading, body) in enumerate(lecture["sections"], start=1):
        nb_cells.append(md_cell(f"## {idx}. {heading}\n\n{body}"))
        if idx == 1:
            nb_cells.append(
                code_cell(
                    f"""
                    lecture = {n}
                    title = {lecture['title_en']!r}
                    source = {lecture['source']!r}
                    print(f"Lecture {{lecture}}: {{title}}")
                    print("source:", source)
                    """
                )
            )

    for idx, (heading, source) in enumerate(lecture["codes"], start=1):
        nb_cells.append(md_cell(f"## 动手实验 {idx}：{heading}\n\n先直接运行，再改一个数字或字符串。"))
        nb_cells.append(code_cell(source))

    checks = "\n".join(f"{i}. {item}" for i, item in enumerate(lecture["checkpoints"], start=1))
    nb_cells.append(md_cell(f"## 今日检查点\n\n{checks}\n\n如果这些能讲清楚，这一讲的第一轮学习就完成了。"))
    return notebook(nb_cells)


def build_markdown(lecture: dict, lang: str) -> str:
    n = lecture["num"]
    filename = f"lecture-{n:02d}-{lecture['slug']}.zh.ipynb"
    if lang == "zh":
        title = f"# 第 {n} 讲：{lecture['title_zh']}\n"
        intro = f"""
        来源：`{lecture['source']}`

        这是 Lecture {n:02d} 的中文文本版。主学习材料是可执行 notebook：

        - `notes/lectures/{filename}`

        这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

        ## 0. 本讲主线

        {lecture['summary_zh']}
        """
        section_lines = []
        for i, (heading, body) in enumerate(lecture["sections"], start=1):
            section_lines.append(f"## {i}. {heading}\n\n对应 notebook：第 {i} 节。\n\n{body}\n")
        checks = "\n".join(f"- {item}" for item in lecture["checkpoints"])
        return title + textwrap.dedent(intro).strip() + "\n\n" + "\n".join(section_lines) + f"\n## 检查点\n\n{checks}\n"

    title = f"# Lecture {n:02d}: {lecture['title_en']}\n"
    intro = f"""
    Source: `{lecture['source']}`

    This is the English text companion. The primary learning artifact is the executable Chinese notebook:

    - `notes/lectures/{filename}`

    This Markdown file is for quick review and terminology lookup.

    ## Main Thread

    {lecture['summary_zh']}
    """
    section_lines = []
    for i, (heading, body) in enumerate(lecture["sections"], start=1):
        section_lines.append(f"## {i}. {heading}\n\nNotebook pointer: section {i}.\n\n{body}\n")
    checks = "\n".join(f"- {item}" for item in lecture["checkpoints"])
    return title + textwrap.dedent(intro).strip() + "\n\n" + "\n".join(section_lines) + f"\n## Checkpoints\n\n{checks}\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lecture in LECTURES:
        n = lecture["num"]
        stem = f"lecture-{n:02d}-{lecture['slug']}"
        (OUT_DIR / f"{stem}.zh.ipynb").write_text(
            json.dumps(build_notebook(lecture), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        (OUT_DIR / f"{stem}.zh.md").write_text(build_markdown(lecture, "zh"), encoding="utf-8")
        (OUT_DIR / f"{stem}.en.md").write_text(build_markdown(lecture, "en"), encoding="utf-8")
        print(f"generated {stem}")

    readme = ["# CS336 Lecture Study Materials\n\n", "主学习材料是 `.zh.ipynb`，Markdown 文件用于快速复习和术语对照。\n\n"]
    readme.append("| Lecture | Notebook | Source |\n| --- | --- | --- |\n")
    readme.append("| 01 | `lecture-01-overview-tokenization.zh.ipynb` | `external/cs336-lectures/lecture_01.py` |\n")
    for lecture in LECTURES:
        n = lecture["num"]
        stem = f"lecture-{n:02d}-{lecture['slug']}"
        readme.append(f"| {n:02d} | `{stem}.zh.ipynb` | `{lecture['source']}` |\n")
    (OUT_DIR / "README.md").write_text("".join(readme), encoding="utf-8")


if __name__ == "__main__":
    main()
