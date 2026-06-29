# Week 06: Training Loop And Debugging

Goal: make a training loop reliable enough to learn from.

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

