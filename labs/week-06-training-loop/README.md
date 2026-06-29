# Week 06：Training Loop 和 Debugging

目标：让训练循环足够可靠，能从实验结果里学东西。

## 这个 Lab 放在哪里

对应 lecture：

- training、optimization、resource accounting、experiment discipline。

对应官方作业：

- Assignment 1：训练一个最小 language model。
- Assignment 3：scaling laws 需要干净的实验记录。

做这个 lab 前：

- 完成 Lab 05。
- 开始完整阅读 Assignment 1。

做完这个 lab 后：

- 认真推进 Assignment 1。
- Assignment 1 基本跑通后，进入 Lab 07，为 Assignment 2/3 做准备。

运行：

```bash
source .venv/bin/activate
python labs/week-06-training-loop/training_loop.py
```

重点观察：

- train loss 和 validation loss 是分开测的。
- optimizer step 前会做 gradient clipping。
- checkpoint 会被保存并加载。
- 每次只改一个 config，实验才容易解释。

问题：

- validation loss 为什么重要？
- gradient clipping 主要防什么问题？
- 为了可靠恢复训练，需要保存哪些东西？
