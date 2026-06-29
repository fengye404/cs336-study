# Week 03: Embeddings, Logits, And LM Objective

Goal: understand next-token prediction.

## Where This Fits

Lecture connection:

- Language modeling objective, embeddings, logits, and cross entropy.

Official assignment connection:

- Assignment 1: Basics, model/loss foundation.

Before this lab:

- Finish Lab 02.
- Know what token ids are and why targets are shifted by one.

After this lab:

- Continue to Lab 04 before doing serious Assignment 1 model work.
- In Assignment 1, identify where logits and cross entropy appear.

Run:

```bash
source .venv/bin/activate
python labs/week-03-lm-objective/train_bigram_lm.py
```

What to look for:

- Input is token ids with shape `(batch, time)`.
- Embedding maps ids to vectors.
- The model produces logits with shape `(batch, time, vocab_size)`.
- Cross entropy compares logits against the next token ids.

Questions:

- Why do targets equal inputs shifted by one position?
- What is a logit?
- Why does cross entropy expect class indices instead of one-hot vectors?
