# Week 05: Tiny GPT

Goal: assemble embeddings, attention, MLP, residuals, and an LM head.

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

