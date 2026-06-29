# Week 07：Scaling 基础和 GPU 显存

目标：在启动训练前，先粗略估算显存。

## 这个 Lab 放在哪里

对应 lecture：

- resource accounting、GPUs/TPUs、systems lectures。

对应官方作业：

- Assignment 2: Systems。
- Assignment 3: Scaling。

做这个 lab 前：

- 完成 Lab 06。
- 至少有一个部分跑通的 Assignment 1 model/training loop。

做完这个 lab 后：

- 开始读 Assignment 2。
- 做 Assignment 3 时，把这些估算放在手边，用来思考 model size、batch size 和 training budget。

运行：

```bash
source .venv/bin/activate
python labs/week-07-memory-estimates/estimate_memory.py
```

重点观察：

- parameters 只是显存的一部分。
- AdamW optimizer states 往往比 parameters 更占空间。
- sequence length 和 batch size 会推高 activation memory。
- 估算不需要完美，但能帮你提前做 sanity check。

问题：

- parameters 占用什么显存？
- gradients 占用什么显存？
- optimizer states 为什么贵？
- activation memory 为什么会随着 sequence length 增长？
