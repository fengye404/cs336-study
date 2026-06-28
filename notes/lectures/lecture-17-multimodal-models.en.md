# Lecture 17: Multimodal Models
Source: `external/cs336-lectures/lecture_17.py`

This is the English text companion. The primary learning artifact is the executable Chinese notebook:

- `notes/lectures/lecture-17-multimodal-models.zh.ipynb`

This Markdown file is for quick review and terminology lookup.

## Main Thread

这一讲讲语言模型如何接入图像、音频、视频等模态：把不同模态转成 token-like 表示，再通过 projector、encoder 或统一架构和语言模型连接。

## 1. 本讲路线图

Notebook pointer: section 1.

多模态输入、视觉 encoder、patch token、CLIP/VLM、统一 token 序列和训练阶段。

## 2. 核心问题

Notebook pointer: section 2.

语言模型吃 token 序列，图像/音频/视频必须先变成可对齐的向量序列。

## 3. Vision Transformer

Notebook pointer: section 3.

把图像切成 patch，每个 patch 类似一个视觉 token，再加位置编码进入 Transformer。

## 4. CLIP 直觉

Notebook pointer: section 4.

用图文对比学习把图像和文本映射到同一个语义空间。

## 5. VLM 连接方式

Notebook pointer: section 5.

常见方法是视觉 encoder + projector + LLM，或者更统一的 multimodal token 模型。

## 6. Token budget

Notebook pointer: section 6.

图像和视频会消耗大量 token/patch，压缩和对齐很重要。

## Checkpoints

- 能解释为什么图像要 patchify。
- 能说出视觉 encoder + projector + LLM 的基本结构。
- 能理解多模态模型为什么容易受 token budget 限制。
