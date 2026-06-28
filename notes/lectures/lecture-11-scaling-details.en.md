# Lecture 11: Scaling Case Study And Details
Source: `external/cs336-lectures/lecture_11.pdf`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-11-scaling-details.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲继续 scaling：实际调参、初始化、optimizer、学习率、batch、参数化方式，以及如何节省拟合 scaling curve 的 compute。

## 1. 本讲路线图

Notebook pointer: section 1.

从 scaling law 到实践：hparam tuning、初始化、optimizer、参数化和跨尺度稳定性。

## 2. Best practice 问题

Notebook pointer: section 2.

大模型不能盲目调参，只能用小规模实验指导大规模设置。

## 3. 初始化与参数化

Notebook pointer: section 3.

不同参数化方式会改变 activation、gradient 和 learning rate 对尺度的敏感度。

## 4. Optimizer 与 learning rate

Notebook pointer: section 4.

AdamW、Muon、batch size、warmup、schedule 都可能随规模变化。

## 5. 节省 compute

Notebook pointer: section 5.

用小模型/短训练拟合趋势，用代理指标筛选，再把少数设置放大验证。

## Checkpoints

- 能解释为什么大模型超参数不能只靠试错。
- 能说出初始化/参数化为什么影响 scaling。
- 能描述一个小实验到大实验的筛选流程。
