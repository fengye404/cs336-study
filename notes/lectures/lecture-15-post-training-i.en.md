# Lecture 15: After Pretraining
Source: `external/cs336-lectures/lecture_15.pdf`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-15-post-training-i.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲讲预训练之后怎么让模型更可控：instruction data、SFT、偏好数据、RLHF/DPO 等 post-training 方法。

## 1. 本讲路线图

Notebook pointer: section 1.

从 GPT-3 式 base model 到 InstructGPT 式可控模型，需要收集目标行为数据并优化模型。

## 2. 为什么预训练不够

Notebook pointer: section 2.

预训练学到 next-token prediction，但用户需要遵循指令、拒绝不当请求、稳定格式和有用回答。

## 3. Instruction data

Notebook pointer: section 3.

SFT 数据通常是 prompt-response 对，直接教模型模仿目标行为。

## 4. Preference data

Notebook pointer: section 4.

偏好数据是同一 prompt 下多个回答的相对好坏，用于 reward model、DPO 或 RLHF。

## 5. SFT 与偏好优化

Notebook pointer: section 5.

SFT 学会格式和基础行为；偏好优化进一步推动更符合人类或规则偏好的输出。

## 6. 规模问题

Notebook pointer: section 6.

post-training 数据更贵、更少，但对模型交互行为影响很大。

## Checkpoints

- 能区分 pretraining、SFT、preference optimization。
- 能描述一条 instruction tuning 数据长什么样。
- 能解释 preference pair 比单个回答多了什么监督信号。
