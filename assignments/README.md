# Assignments

这里用 Git submodule 跟踪 CS336 官方 assignment repo。

## 和 Labs 的关系

官方作业才是真正的课程项目。本仓库里的 labs 是热身和概念拆解，用来降低直接写官方作业时的认知负担。

推荐时间点：

| Assignment | 做完什么后开始读 | 做完什么后开始实现 |
| --- | --- | --- |
| Assignment 1: Basics | Lab 02 | Lab 05 or 06 |
| Assignment 2: Systems | Lab 07 | Lab 08 |
| Assignment 3: Scaling | Lab 06 | Lab 07 or 08 |
| Assignment 4: Data | Lab 09 | 官方 data lectures 13-14 |
| Assignment 5: Alignment | Lab 10 | SFT / preference 概念清楚之后 |

不要把 labs 当成答案。它们的作用是帮你在读官方 handout 前，先掌握术语、shape 和最小实现。

## 当前官方作业

- `assignment1-basics`：tokenization、Transformer 基础、optimizer、training loop。
- `assignment2-systems`：profiling、kernels、FlashAttention、distributed/system work。
- `assignment3-scaling`：scaling laws 和 scaling experiments。
- `assignment4-data`：data processing、filtering、deduplication、curation。
- `assignment5-alignment`：post-training、alignment、reasoning RL。

在新机器上 clone 这个学习仓库后，用下面命令初始化官方作业 submodule：

```bash
git submodule update --init --recursive
```

官方作业代码写在对应 submodule 目录里；学习笔记、复盘和自己的理解放在这个 study repo 里。
