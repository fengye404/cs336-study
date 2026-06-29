# Week 09: Inference, Sampling, And Evaluation

Goal: connect logits to generated text and simple evaluation.

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

