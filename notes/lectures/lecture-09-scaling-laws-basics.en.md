# Lecture 09: Scaling Laws Basics
Source: `external/cs336-lectures/lecture_09.pdf`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-09-scaling-laws-basics.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲讲如何认真对待 scaling：用小实验拟合简单预测规律，再外推到大模型，帮助选择模型大小、数据量和 compute。

## 1. 本讲路线图

Notebook pointer: section 1.

为什么需要 scaling laws，如何拟合，如何用它们做预算决策。

## 2. 问题设定

Notebook pointer: section 2.

给定固定训练预算，应该训练多大的模型、用多少数据、跑多久？

## 3. Power law 直觉

Notebook pointer: section 3.

很多 loss 随 compute、参数、数据变化呈近似幂律关系，在 log-log 坐标里接近直线。

## 4. Chinchilla 风格权衡

Notebook pointer: section 4.

模型太大数据太少会 undertrain，模型太小数据太多也浪费。

## 5. 外推的风险

Notebook pointer: section 5.

小规模规律不一定完全延伸到大规模，数据质量、架构、优化都会改变曲线。

## Checkpoints

- 能解释为什么 log-log 图上直线代表 power law。
- 能用 scaling law 的思路比较两个训练方案。
- 能说出外推为什么有风险。
