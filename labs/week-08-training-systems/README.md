# Week 08：高效训练和分布式概念

目标：把训练系统技术和它们解决的 bottleneck 对上号。

## 这个 Lab 放在哪里

对应 lecture：

- kernels、Triton、parallelism、distributed training。

对应官方作业：

- Assignment 2: Systems 是最直接对应。

做这个 lab 前：

- 完成 Lab 07。
- 回头复习 Lab 04 的 attention shapes。

做完这个 lab 后：

- 如果你准备好写 systems-heavy code，可以开始 Assignment 2。
- 如果 systems 感觉陡，先完成 Assignment 1，再只读 Assignment 2 handout，不急着实现。

运行：

```bash
source .venv/bin/activate
python labs/week-08-training-systems/bottleneck_map.py
```

重点观察：

- data parallelism 通过在多个 worker 上复制模型来提升 throughput。
- tensor parallelism 拆分大型矩阵计算。
- pipeline parallelism 拆分模型层。
- FSDP/ZeRO 会 shard parameters、gradients 和 optimizer states。
- FlashAttention 减少 attention 的 memory traffic。

问题：

- 哪些技术主要降低 memory？
- 哪些技术主要提升 throughput？
- 哪些技术会增加 communication？
- 如果 attention activations 是 bottleneck，应该看哪类技术？
