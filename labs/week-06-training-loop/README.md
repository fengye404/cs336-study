# Week 06: Training Loop And Debugging

Goal: make a training loop reliable enough to learn from.

## Where This Fits

Lecture connection:

- Training, optimization, resource accounting, and experiment discipline.

Official assignment connection:

- Assignment 1: training a minimal language model.
- Assignment 3: scaling laws later depend on clean experiment logs.

Before this lab:

- Finish Lab 05.
- Start reading Assignment 1 end to end.

After this lab:

- Work seriously on Assignment 1.
- When Assignment 1 is mostly working, move to Lab 07 and prepare for Assignment 2/3.

Run:

```bash
source .venv/bin/activate
python labs/week-06-training-loop/training_loop.py
```

What to look for:

- Train and validation loss are measured separately.
- Gradients are clipped before the optimizer step.
- A checkpoint is saved and loaded.
- Changing one config value at a time makes experiments easier to reason about.

Questions:

- Why does validation loss matter?
- What problem does gradient clipping prevent?
- What needs to be saved to resume training faithfully?
