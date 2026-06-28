# 第 11 讲：Scaling 实践与细节
来源：`external/cs336-lectures/lecture_11.pdf`

这是 Lecture 11 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-11-scaling-details.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲继续 scaling：实际调参、初始化、optimizer、学习率、batch、参数化方式，以及如何节省拟合 scaling curve 的 compute。

## 1. 本讲路线图

对应 notebook：第 1 节。

从 scaling law 到实践：hparam tuning、初始化、optimizer、参数化和跨尺度稳定性。

## 2. Best practice 问题

对应 notebook：第 2 节。

大模型不能盲目调参，只能用小规模实验指导大规模设置。

## 3. 初始化与参数化

对应 notebook：第 3 节。

不同参数化方式会改变 activation、gradient 和 learning rate 对尺度的敏感度。

## 4. Optimizer 与 learning rate

对应 notebook：第 4 节。

AdamW、Muon、batch size、warmup、schedule 都可能随规模变化。

## 5. 节省 compute

对应 notebook：第 5 节。

用小模型/短训练拟合趋势，用代理指标筛选，再把少数设置放大验证。

## 检查点

- 能解释为什么大模型超参数不能只靠试错。
- 能说出初始化/参数化为什么影响 scaling。
- 能描述一个小实验到大实验的筛选流程。
