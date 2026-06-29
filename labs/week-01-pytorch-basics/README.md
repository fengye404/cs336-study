# Week 01: PyTorch Basics

Goal: run and understand one complete training loop.

## Where This Fits

Official CS336 assumes Python, PyTorch, and systems comfort. This lab is a personal warmup before the official material starts moving quickly.

Before this lab:

- Skim the CS336 homepage and prerequisites.
- Read the Week 1 section in `study-plan.md`.

After this lab:

- Continue to Lab 02.
- Do not start official Assignment 1 yet unless the training loop feels readable.

Run:

```bash
source .venv/bin/activate
python labs/week-01-pytorch-basics/train_tiny_mlp.py
```

What to look for:

- Tensor shapes printed near the beginning.
- Training loss should go down.
- Validation loss should be reasonable.
- The model should learn the synthetic function better than random guessing.

Questions to answer in `notes/week-01-setup.md`:

- What is the difference between a tensor and a NumPy array?
- What does `loss.backward()` compute?
- Why do we call `optimizer.zero_grad()` before backprop?
- Which tensor shapes flow through the model?
