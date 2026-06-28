# Lecture 16: Post-training II: RL From Verifiable Rewards
Source: `external/cs336-lectures/lecture_16.pdf`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-16-post-training-ii-rlvr.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲从 RLHF 走到可验证奖励：在数学、代码等领域可以用明确 reward 训练长推理行为，涉及 PPO、GRPO 和 long-CoT 现象。

## 1. 本讲路线图

Notebook pointer: section 1.

核心算法、案例研究、现象：PPO 到 GRPO，可验证奖励，长 CoT，SFT vs RL。

## 2. 为什么可验证奖励重要

Notebook pointer: section 2.

如果答案能自动验证，就不完全依赖人类偏好模型，RL 信号更清楚也更容易扩展。

## 3. PPO / GRPO 直觉

Notebook pointer: section 3.

策略生成多个回答，根据 reward 调整概率，同时限制模型不要偏离太远。

## 4. Long-CoT

Notebook pointer: section 4.

RL 可能鼓励模型生成更长的推理轨迹，只要这些轨迹提高可验证任务得分。

## 5. SFT vs RL

Notebook pointer: section 5.

SFT 模仿已有答案；RL 可以探索新策略，但更不稳定。

## 6. 系统问题

Notebook pointer: section 6.

RL 需要大量采样、打分和训练循环，inference 成本也成为训练成本的一部分。

## Checkpoints

- 能解释 verifiable reward 和 human preference 的区别。
- 能说出 GRPO 的 group-relative 信号直觉。
- 能理解为什么 RL 训练会增加 inference 成本。
