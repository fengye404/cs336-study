# Week 04: Causal Self-Attention

Goal: implement attention and inspect every shape.

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

