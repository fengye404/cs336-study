# Week 05: Tiny GPT

Goal: assemble embeddings, attention, MLP, residuals, and an LM head.

## Where This Fits

Lecture connection:

- Transformer architecture, hyperparameters, normalization, attention, and MLPs.

Official assignment connection:

- Assignment 1: Basics, full model architecture.

Before this lab:

- Finish Labs 02-04.
- Know tokenization, logits, attention, and causal masking at a high level.

After this lab:

- Read Assignment 1 carefully.
- Start implementing Assignment 1 if you can explain this lab's forward pass.
- Continue to Lab 06 in parallel with Assignment 1 training/debugging.

Run:

```bash
source .venv/bin/activate
python labs/week-05-tiny-gpt/tiny_gpt.py
```

What to look for:

- Token and positional embeddings are added.
- Each Transformer block keeps shape `(batch, time, channels)`.
- The LM head maps channels to vocabulary logits.
- Generated samples improve a little after training.

Questions:

- Why do residual connections preserve shape?
- Why does the model need positional embeddings?
- What changes between the Week 03 bigram model and this tiny GPT?
