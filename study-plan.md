# Study Plan

This plan adapts Stanford CS336: Language Modeling from Scratch for a working schedule:

- Work starts at 10:00.
- Work ends around 22:00.
- Weekdays are for light input, review, and small code changes.
- Weekends are for deep implementation.

The official CS336 pace is roughly a 10-week Stanford course with heavy assignments. For this repo, the plan stretches it to 18 weeks so it can survive real workdays.

## Weekly Rhythm

Use this as the default schedule.

| Time | Task | Energy |
| --- | --- | --- |
| Monday 08:30-09:15 | Preview the week's lecture/theme | Light |
| Wednesday 08:30-09:15 | Read notes or watch part of a lecture | Light |
| Friday 22:30-23:00 | Write questions and plan weekend work | Very light |
| Saturday 10:00-12:00 | Main coding block | Deep |
| Saturday 14:30-17:30 | Assignment/lab implementation | Deep |
| Sunday 10:00-12:00 | Debugging and cleanup | Medium |
| Sunday 20:30-21:15 | Weekly checkpoint | Light |

Expected load: 7-9 hours/week.

Minimum viable week when work is brutal:

- Watch or read 45 minutes.
- Write one note in `notes/`.
- Make one tiny code change or run one experiment.
- Add a checkpoint with what blocked progress.

## Rules

- Do not study after 23:30 unless it is genuinely fun that day.
- Do not start a new concept on Sunday night.
- For each topic, produce one artifact: note, lab, assignment commit, or checkpoint.
- Prefer understanding tensor shapes over copying working code.
- When stuck for more than 45 minutes, write the exact confusion in `notes/` before searching.

## Phase 0: Setup And Python/PyTorch Warmup

Duration: 2 weeks

Goal: be able to read and modify CS336 assignment code.

### Week 1: Python For Training Code

- Set up `.venv`.
- Review Python syntax, list/dict comprehensions, classes, iterators, and file I/O.
- Learn NumPy arrays, broadcasting, matrix multiplication, and indexing.
- Lab: implement a tiny linear regression with NumPy.

Done when:

- I can read Python code without translating everything back to Java.
- I can explain array shapes while doing matrix multiplication.

### Week 2: PyTorch Basics

- Learn tensors, autograd, `nn.Module`, optimizers, and dataloaders.
- Train a tiny MLP on synthetic data.
- Lab: print tensor shapes at every step of a forward/backward pass.

Done when:

- I can train a tiny model and explain what `loss.backward()` does.

## Phase 1: Tokenization And Data

Duration: 2 weeks

Maps to CS336 topics around data processing and tokenization.

### Week 3: Unicode, Bytes, And BPE

- Study Unicode, UTF-8, bytes, vocabulary, merges, and special tokens.
- Read CS336 tokenization material.
- Lab: compare token counts for real agent prompts.

Done when:

- I can explain why the same text can have different token counts under different tokenizers.

### Week 4: Tokenizer Implementation

- Trace or implement a small BPE tokenizer.
- Read the official assignment structure before coding.
- Lab: build a small tokenizer on a toy corpus.

Done when:

- I can explain training-time tokenizer construction and inference-time encoding.

## Phase 2: Transformer Core

Duration: 4 weeks

Maps to CS336's Transformer and language-modeling core.

### Week 5: Embeddings And Language Modeling Objective

- Study token embeddings, positional information, logits, softmax, and cross entropy.
- Lab: build a minimal next-token prediction loop.

Done when:

- I can explain one training example from input token ids to loss.

### Week 6: Attention

- Study scaled dot-product attention, masks, and multi-head attention.
- Lab: implement attention with explicit tensor shape comments.

Done when:

- I can explain `Q`, `K`, `V`, attention scores, causal mask, and output shape.

### Week 7: Transformer Block

- Study residual connections, layer norm/RMSNorm, MLP, activations, and dropout.
- Lab: implement one Transformer block.

Done when:

- I can trace shapes through a full block without guessing.

### Week 8: Tiny GPT

- Combine tokenizer, embeddings, Transformer blocks, and LM head.
- Train a tiny character-level or token-level model.
- Add loss curve and generated samples.

Done when:

- I can explain a full forward pass from text to next-token logits.

## Phase 3: Training Mechanics

Duration: 3 weeks

Maps to CS336 topics around optimization, training loops, and experiments.

### Week 9: Training Loop

- Study batching, gradient accumulation, clipping, scheduler, and checkpointing.
- Lab: add checkpoint save/load to the tiny model.

Done when:

- I can stop and resume a toy training run.

### Week 10: Optimization And Debugging

- Study AdamW, learning-rate schedules, initialization, and loss curves.
- Lab: run 3 small experiments changing one variable at a time.

Done when:

- I can diagnose obvious issues like no learning, exploding loss, or bad batch shapes.

### Week 11: Data And Experiment Hygiene

- Study train/val split, overfitting, reproducibility, seeds, and logging.
- Lab: add validation loss and a simple experiment log.

Done when:

- I can compare two runs without relying on vibes.

## Phase 4: Scaling And Systems

Duration: 3 weeks

Maps to CS336's systems-heavy parts.

### Week 12: GPU Memory And Throughput

- Study parameters, activations, optimizer states, batch size, and sequence length.
- Lab: estimate memory usage for several toy model sizes.

Done when:

- I can do a rough memory estimate before running code.

### Week 13: Mixed Precision And Efficient Attention

- Study fp32, fp16, bf16, FlashAttention, and numerical stability.
- Lab: compare a small run with and without mixed precision if hardware allows.

Done when:

- I can explain why training speed and stability trade off.

### Week 14: Distributed Training Concepts

- Study data parallelism, tensor parallelism, pipeline parallelism, and FSDP.
- No need to implement full distributed training unless it becomes useful.

Done when:

- I can read a distributed-training article without losing the plot.

## Phase 5: Inference And Evaluation

Duration: 2 weeks

Maps to CS336 inference, generation, and evaluation topics.

### Week 15: Decoding And KV Cache

- Study greedy decoding, temperature, top-k, top-p, prefill, decode, and KV cache.
- Lab: compare generated outputs under different sampling settings.

Done when:

- I can connect inference settings to agent behavior.

### Week 16: Evaluation

- Study perplexity, benchmark design, contamination, and qualitative evals.
- Lab: create a tiny eval set for agent-style prompts.

Done when:

- I can explain why "the model feels better" is not enough.

## Phase 6: Alignment Bridge

Duration: 2 weeks

This is where CS336 connects back to agent development, RLHF, and preference learning.

### Week 17: SFT And Preference Data

- Study supervised fine-tuning, instruction data, preference pairs, and reward models.
- Write a note mapping agent traces to possible training data.

Done when:

- I can identify which parts of an agent workflow could become SFT data or preference data.

### Week 18: RLHF, DPO, And Next Plan

- Study PPO at a high level, DPO, and why modern alignment is not just "RL on text."
- Write a final checkpoint: what I now understand, what to learn next.

Done when:

- I can explain where RL fundamentals matter for LLM alignment and where preference optimization is enough.

## Weekly Checkpoint Template

Create one file under `checkpoints/`, for example `checkpoints/week-01.md`.

```markdown
# Week 01

## Time Spent

- Weekday:
- Weekend:

## Finished

- ...

## Can Explain

- ...

## Can Implement

- ...

## Still Confusing

- ...

## Next Week

- ...
```

## First Month Success Criteria

After four weeks, I should have:

- A working Python environment.
- A tiny NumPy or PyTorch training script.
- Notes on tokenization.
- A small tokenizer lab or assignment attempt.
- Four weekly checkpoints.

That is enough momentum. The goal is not speed; the goal is not silently quitting.
