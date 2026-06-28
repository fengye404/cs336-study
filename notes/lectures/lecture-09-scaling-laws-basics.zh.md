# 第 9 讲：Scaling Laws 基础
来源：`external/cs336-lectures/lecture_09.pdf`

这是 Lecture 09 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-09-scaling-laws-basics.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲讲如何认真对待 scaling：用小实验拟合简单预测规律，再外推到大模型，帮助选择模型大小、数据量和 compute。

## 1. 本讲路线图

对应 notebook：第 1 节。

为什么需要 scaling laws，如何拟合，如何用它们做预算决策。

## 2. 问题设定

对应 notebook：第 2 节。

给定固定训练预算，应该训练多大的模型、用多少数据、跑多久？

## 3. Power law 直觉

对应 notebook：第 3 节。

很多 loss 随 compute、参数、数据变化呈近似幂律关系，在 log-log 坐标里接近直线。

## 4. Chinchilla 风格权衡

对应 notebook：第 4 节。

模型太大数据太少会 undertrain，模型太小数据太多也浪费。

## 5. 外推的风险

对应 notebook：第 5 节。

小规模规律不一定完全延伸到大规模，数据质量、架构、优化都会改变曲线。

## 检查点

- 能解释为什么 log-log 图上直线代表 power law。
- 能用 scaling law 的思路比较两个训练方案。
- 能说出外推为什么有风险。
