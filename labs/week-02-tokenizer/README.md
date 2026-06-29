# Week 02: Tokenization And BPE

Goal: understand how text becomes token ids.

## Where This Fits

Lecture connection:

- CS336 overview and tokenization material.

Official assignment connection:

- Warmup for Assignment 1: Basics, tokenizer portion.

Before this lab:

- Finish Lab 01.
- Skim the official Assignment 1 tokenizer section.

After this lab:

- Continue to Lab 03.
- You can read Assignment 1 tests, but wait until Lab 05 or 06 before trying to finish it.

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
