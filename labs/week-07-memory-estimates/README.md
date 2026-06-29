# Week 07: Scaling Basics And GPU Memory

Goal: estimate memory before launching a run.

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

