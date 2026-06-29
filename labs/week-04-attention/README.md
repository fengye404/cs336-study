# Week 04: Causal Self-Attention

Goal: implement attention and inspect every shape.

## Where This Fits

Lecture connection:

- Transformer architecture, attention, and attention alternatives.

Official assignment connection:

- Assignment 1: implement Transformer components.
- Assignment 2: later optimize attention and understand FlashAttention-style constraints.

Before this lab:

- Finish Lab 03.
- Be comfortable with tensors shaped like `(batch, time, channels)`.

After this lab:

- Continue to Lab 05.
- Revisit this lab before Assignment 2, because systems work starts by knowing attention's memory and compute pattern.

Run:

```bash
source .venv/bin/activate
python labs/week-04-attention/attention_shapes.py
```

What to look for:

- `Q`, `K`, and `V` all come from the same input in self-attention.
- Attention scores have shape `(batch, heads, time, time)`.
- A causal mask blocks tokens from seeing the future.
- Attention output returns to shape `(batch, time, channels)`.

Questions:

- Why do we divide attention scores by `sqrt(head_dim)`?
- Why does causal language modeling need a lower-triangular mask?
- Which dimensions are sequence length, embedding size, and number of heads?
