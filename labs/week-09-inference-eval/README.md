# Week 09: Inference, Sampling, And Evaluation

Goal: connect logits to generated text and simple evaluation.

## Where This Fits

Lecture connection:

- Inference, evaluation, and parts of the data section.

Official assignment connection:

- Assignment 2: inference/system benchmarking concepts.
- Assignment 4: Data and evaluation, though this lab only covers a small slice.

Before this lab:

- Finish Labs 03-06.
- Understand logits and cross entropy.

After this lab:

- For Assignment 2, connect sampling/inference to runtime constraints.
- For Assignment 4, watch/read the official data lectures; this repo does not replace the data-cleaning assignment.

Run:

```bash
source .venv/bin/activate
python labs/week-09-inference-eval/sampling_and_eval.py
```

What to look for:

- Greedy decoding always takes the highest-probability token.
- Temperature changes confidence.
- Top-k and top-p restrict the candidate set.
- Perplexity is exponentiated average negative log-likelihood.

Questions:

- Why can lower temperature make agent behavior more stable?
- Why can top-p preserve more flexibility than top-k?
- Why is one eval set never enough?
