# Labs

Small runnable experiments live here. These are personal learning labs, not official CS336 assignments.

## Relationship To CS336

CS336 has three layers:

- Lectures: explain the ideas and design tradeoffs.
- These labs: small runnable drills that make one idea concrete.
- Official assignments: larger implementation projects with much less scaffolding.

Use the order:

1. Watch or skim the relevant lecture material.
2. Run and understand the matching lab.
3. Start the official assignment once several related labs feel familiar.

These labs are intentionally smaller and more guided than the official assignments. They are allowed to use PyTorch directly, while official assignments may ask you to implement lower-level pieces yourself.

Run them from the repository root:

```bash
source .venv/bin/activate
python labs/week-01-pytorch-basics/train_tiny_mlp.py
```

## Map

| Week | Lab | Purpose |
| --- | --- | --- |
| 01 | `week-01-pytorch-basics` | PyTorch training loop |
| 02 | `week-02-tokenizer` | Toy BPE and token counting |
| 03 | `week-03-lm-objective` | Embeddings, logits, cross entropy |
| 04 | `week-04-attention` | Causal self-attention shapes |
| 05 | `week-05-tiny-gpt` | Minimal GPT-style model |
| 06 | `week-06-training-loop` | Validation, checkpointing, clipping |
| 07 | `week-07-memory-estimates` | Rough memory estimates |
| 08 | `week-08-training-systems` | Distributed-training concept map |
| 09 | `week-09-inference-eval` | Sampling and perplexity |
| 10 | `week-10-alignment-bridge` | Preference data and DPO loss |

Each lab should produce one note or checkpoint. The point is not to perfect the code; the point is to make the concept executable.

## Official Assignment Timing

| Official work | Do after | Why |
| --- | --- | --- |
| Assignment 1: Basics | Labs 02-06 | Tokenizer, model architecture, objective, attention, optimizer/training loop |
| Assignment 2: Systems | Labs 04, 06, 07, 08, 09 | Attention, benchmarking mindset, memory accounting, parallelism, inference |
| Assignment 3: Scaling | Labs 06-08 | Experiment hygiene, memory/FLOP estimates, scaling intuition |
| Assignment 4: Data | Lab 09 plus CS336 data lectures | This repo only lightly covers eval/data; use official lectures 13-14 heavily |
| Assignment 5: Alignment | Lab 10 | SFT, preference data, RLHF/DPO bridge |

Fast-track rule: after Lab 05, clone/read Assignment 1 seriously. After Lab 06, begin implementing it seriously.

## Lecture-To-Lab Route

Use this when deciding what to study next.

| Study step | First read/watch | Then run | Then attempt |
| --- | --- | --- | --- |
| Warmup | CS336 prerequisites and Lecture 1 overview | Lab 01 | No official assignment yet |
| Tokenization | Lecture 1 tokenization section | Lab 02 | Assignment 1 tokenizer tests |
| LM objective | Transformer/language-modeling material | Lab 03 | Assignment 1 model/loss code |
| Attention | Transformer attention material | Lab 04 | Assignment 1 attention/model tests |
| Tiny GPT | Transformer architecture material | Lab 05 | Assignment 1 full model implementation |
| Training | Optimization/training-loop material | Lab 06 | Assignment 1 training and optimizer pieces |
| Resource accounting | Lecture 2-style PyTorch/resource accounting material | Lab 07 | Assignment 2 and Assignment 3 handouts |
| Systems | Lecture 7 parallelism and systems material | Lab 08 | Assignment 2 implementation |
| Inference/eval/data | Lecture 10 inference, Lecture 12 evaluation, Lectures 13-14 data | Lab 09 | Assignment 4 after data lectures |
| Alignment | Alignment/RLHF material | Lab 10 | Assignment 5 |

If a lecture feels abstract, run the lab first and come back. If a lab feels like magic, go back to the lecture and write down the missing concept.
