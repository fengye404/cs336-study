# Week 10：Alignment 桥接

目标：把 LLM training 和 SFT、preferences、RLHF、DPO 连起来。

## 这个 Lab 放在哪里

对应 lecture：

- mid/post-training、SFT、RLHF/RLVR、alignment。

对应官方作业：

- Assignment 5：Alignment 和 reasoning RL。

做这个 lab 前：

- 完成 Lab 09。
- 从 language-modeling objective 里理解 log-probabilities 是什么。

做完这个 lab 后：

- 能解释 SFT data 和 preference data 的区别后，再开始 Assignment 5。
- 如果 PPO/RL 还很模糊，先单独补 RL basics，再深入 RLHF。

运行：

```bash
source .venv/bin/activate
python labs/week-10-alignment-bridge/preference_loss.py
```

重点观察：

- preference data 比较 chosen answer 和 rejected answer。
- DPO 会用 policy log-probabilities 和 reference log-probabilities。
- 这个 loss 会奖励 policy 相比 reference 更偏好 chosen responses。

问题：

- agent trace 可以提供哪些 SFT 数据？
- 哪些数据需要人或模型给 preference labels？
- 为什么 DPO 不等于一般意义上的 RL？
