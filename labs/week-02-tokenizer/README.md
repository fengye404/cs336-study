# Week 02: Tokenization And BPE

Goal: understand how text becomes token ids.

Run:

```bash
source .venv/bin/activate
python labs/week-02-tokenizer/toy_bpe.py
python labs/week-02-tokenizer/token_count.py
```

What to look for:

- BPE starts from small pieces and repeatedly merges frequent adjacent pairs.
- Training a tokenizer creates merge rules.
- Encoding text applies existing merge rules.
- Token counts change when wording, punctuation, whitespace, or language changes.

Questions:

- What is a vocabulary?
- What is a merge rule?
- Why does `"hello world"` not necessarily mean two tokens?
- Why does tokenization matter for agent cost and context length?

