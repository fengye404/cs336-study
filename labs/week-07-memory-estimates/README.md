# Week 07: Scaling Basics And GPU Memory

Goal: estimate memory before launching a run.

## Where This Fits

Lecture connection:

- Resource accounting, GPUs/TPUs, and systems lectures.

Official assignment connection:

- Assignment 2: Systems.
- Assignment 3: Scaling.

Before this lab:

- Finish Lab 06.
- Have at least a partial Assignment 1 model/training loop.

After this lab:

- Start reading Assignment 2.
- For Assignment 3, keep these estimates nearby when thinking about model size, batch size, and training budget.

Run:

```bash
source .venv/bin/activate
python labs/week-07-memory-estimates/estimate_memory.py
```

What to look for:

- Parameters are only one part of memory.
- AdamW optimizer states often cost more than parameters.
- Sequence length and batch size drive activation memory.
- Estimates are approximate but useful for sanity checks.

Questions:

- What memory is used by parameters?
- What memory is used by gradients?
- Why are optimizer states expensive?
- Why does activation memory grow with sequence length?
