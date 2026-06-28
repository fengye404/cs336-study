# Lecture 13: Data I
Source: `external/cs336-lectures/lecture_13.py`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-13-data-i.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲进入数据：数据来源、法律与许可、爬取、过滤、DCLM/CommonCrawl 这类数据 pipeline 的目标和风险。

## 1. 本讲路线图

Notebook pointer: section 1.

从数据来源和法律问题开始，到自动构建大规模高质量预训练数据集。

## 2. 数据为什么重要

Notebook pointer: section 2.

pretraining loss 和能力很大程度取决于 token 来自哪里、质量如何、比例如何。

## 3. 法律与许可

Notebook pointer: section 3.

公开可访问不等于可自由使用；license、robots、copyright 都影响数据策略。

## 4. Crawling 与 extraction

Notebook pointer: section 4.

网页需要抓取、抽正文、去模板、去广告、去噪声。

## 5. Filtering

Notebook pointer: section 5.

规则过滤、质量模型过滤、语言过滤、安全过滤会改变数据分布。

## 6. 数据集是系统

Notebook pointer: section 6.

数据 pipeline 要可复现、可审计、可迭代，而不是一次性脚本。

## Checkpoints

- 能解释为什么数据质量不是单一分数。
- 能说出过滤会改变数据分布。
- 能理解 license 和来源记录为什么重要。
