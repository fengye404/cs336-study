# 第 17 讲：多模态模型
来源：`external/cs336-lectures/lecture_17.py`

这是 Lecture 17 的中文文本版。主学习材料是可执行 notebook：

- `notes/lectures/lecture-17-multimodal-models.zh.ipynb`

这个 Markdown 适合快速复习；notebook 按相同结构加入了可运行实验。

## 0. 本讲主线

这一讲讲语言模型如何接入图像、音频、视频等模态：把不同模态转成 token-like 表示，再通过 projector、encoder 或统一架构和语言模型连接。

## 1. 本讲路线图

对应 notebook：第 1 节。

多模态输入、视觉 encoder、patch token、CLIP/VLM、统一 token 序列和训练阶段。

## 2. 核心问题

对应 notebook：第 2 节。

语言模型吃 token 序列，图像/音频/视频必须先变成可对齐的向量序列。

## 3. Vision Transformer

对应 notebook：第 3 节。

把图像切成 patch，每个 patch 类似一个视觉 token，再加位置编码进入 Transformer。

## 4. CLIP 直觉

对应 notebook：第 4 节。

用图文对比学习把图像和文本映射到同一个语义空间。

## 5. VLM 连接方式

对应 notebook：第 5 节。

常见方法是视觉 encoder + projector + LLM，或者更统一的 multimodal token 模型。

## 6. Token budget

对应 notebook：第 6 节。

图像和视频会消耗大量 token/patch，压缩和对齐很重要。

## 检查点

- 能解释为什么图像要 patchify。
- 能说出视觉 encoder + projector + LLM 的基本结构。
- 能理解多模态模型为什么容易受 token budget 限制。
