# Assignments

Official CS336 assignment repos are tracked here as Git submodules.

## Relationship To Labs

The official assignments are the real course projects. The labs in this repo are warmups and concept drills.

Recommended timing:

| Assignment | Start reading after | Start implementing after |
| --- | --- | --- |
| Assignment 1: Basics | Lab 02 | Lab 05 or 06 |
| Assignment 2: Systems | Lab 07 | Lab 08 |
| Assignment 3: Scaling | Lab 06 | Lab 07 or 08 |
| Assignment 4: Data | Lab 09 | Official data lectures 13-14 |
| Assignment 5: Alignment | Lab 10 | After SFT/preference concepts are clear |

Do not treat the labs as solutions. Treat them as vocabulary and shape practice before reading the official handouts.

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
