# 第 13 讲：Data I
来源：`external/cs336-lectures/lecture_13.py`

这是 Lecture 13 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-13-data-i.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲进入数据：数据来源、法律与许可、爬取、过滤、DCLM/CommonCrawl 这类数据 pipeline 的目标和风险。

## 1. 本讲路线图

对应 notebook：第 1 节。

从数据来源和法律问题开始，到自动构建大规模高质量预训练数据集。

## 2. 数据为什么重要

对应 notebook：第 2 节。

pretraining loss 和能力很大程度取决于 token 来自哪里、质量如何、比例如何。

## 3. 法律与许可

对应 notebook：第 3 节。

公开可访问不等于可自由使用；license、robots、copyright 都影响数据策略。

## 4. Crawling 与 extraction

对应 notebook：第 4 节。

网页需要抓取、抽正文、去模板、去广告、去噪声。

## 5. Filtering

对应 notebook：第 5 节。

规则过滤、质量模型过滤、语言过滤、安全过滤会改变数据分布。

## 6. 数据集是系统

对应 notebook：第 6 节。

数据 pipeline 要可复现、可审计、可迭代，而不是一次性脚本。

## 检查点

- 能解释为什么数据质量不是单一分数。
- 能说出过滤会改变数据分布。
- 能理解 license 和来源记录为什么重要。
