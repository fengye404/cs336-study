# Week 03: Embeddings, Logits, And LM Objective

Goal: understand next-token prediction.

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

