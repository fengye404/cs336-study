# 第 12 讲：Evaluation
来源：`external/cs336-lectures/lecture_12.py`

这是 Lecture 12 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-12-evaluation.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲讨论 evaluation 的目标、对象、数据质量和污染问题。评估不是跑几个 benchmark，而是定义你真正关心的能力和失效模式。

## 1. 本讲路线图

对应 notebook：第 1 节。

评估什么、怎么评估、数据质量、train-test overlap、benchmark 的局限。

## 2. 为什么 evaluation 难

对应 notebook：第 2 节。

模型能力是多维的；benchmark 容易被污染、过拟合或测不到真实使用场景。

## 3. 评估对象

对应 notebook：第 3 节。

可能评估 base LM、chat model、agent、tool-use system，也可能评估 latency/cost/safety。

## 4. Train-test overlap

对应 notebook：第 4 节。

训练数据如果包含测试题，分数会虚高，无法反映泛化。

## 5. 数据质量

对应 notebook：第 5 节。

题目是否清楚、标注是否一致、答案是否唯一，都会影响评估信号。

## 6. 评估的用途

对应 notebook：第 6 节。

选择模型、发现失效模式、监控回归、指导数据和训练改进。

## 检查点

- 能解释 benchmark 分数为什么不是模型质量本身。
- 能说出 train-test overlap 的危害。
- 能设计一个小的 agent 场景 eval。
