# Assignments

Official CS336 assignment repos are tracked here as Git submodules.

## Current Assignments

- `assignment1-basics`: tokenization, Transformer basics, optimizer, training loop.
- `assignment2-systems`: profiling, kernels, FlashAttention, distributed/system work.
- `assignment3-scaling`: scaling laws and scaling experiments.
- `assignment4-data`: data processing, filtering, deduplication, curation.
- `assignment5-alignment`: post-training, alignment, reasoning RL.

After cloning this study repo on a new machine, initialize the official assignment repos with:

```bash
git submodule update --init --recursive
```

Do assignment work inside the corresponding submodule directory, while keeping notes and retrospectives in this study repo.
