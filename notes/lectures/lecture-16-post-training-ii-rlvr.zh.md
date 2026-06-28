# 第 16 讲：Post-training II：可验证奖励 RL
来源：`external/cs336-lectures/lecture_16.pdf`

这是 Lecture 16 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-16-post-training-ii-rlvr.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲从 RLHF 走到可验证奖励：在数学、代码等领域可以用明确 reward 训练长推理行为，涉及 PPO、GRPO 和 long-CoT 现象。

## 1. 本讲路线图

对应 notebook：第 1 节。

核心算法、案例研究、现象：PPO 到 GRPO，可验证奖励，长 CoT，SFT vs RL。

## 2. 为什么可验证奖励重要

对应 notebook：第 2 节。

如果答案能自动验证，就不完全依赖人类偏好模型，RL 信号更清楚也更容易扩展。

## 3. PPO / GRPO 直觉

对应 notebook：第 3 节。

策略生成多个回答，根据 reward 调整概率，同时限制模型不要偏离太远。

## 4. Long-CoT

对应 notebook：第 4 节。

RL 可能鼓励模型生成更长的推理轨迹，只要这些轨迹提高可验证任务得分。

## 5. SFT vs RL

对应 notebook：第 5 节。

SFT 模仿已有答案；RL 可以探索新策略，但更不稳定。

## 6. 系统问题

对应 notebook：第 6 节。

RL 需要大量采样、打分和训练循环，inference 成本也成为训练成本的一部分。

## 检查点

- 能解释 verifiable reward 和 human preference 的区别。
- 能说出 GRPO 的 group-relative 信号直觉。
- 能理解为什么 RL 训练会增加 inference 成本。
