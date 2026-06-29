# Week 08: Efficient Training And Distributed Concepts

Goal: map training-system techniques to the bottlenecks they address.

## Where This Fits

Lecture connection:

- Kernels, Triton, parallelism, and distributed training.

Official assignment connection:

- Assignment 2: Systems is the main match.

Before this lab:

- Finish Lab 07.
- Revisit Lab 04 attention shapes.

After this lab:

- Work on Assignment 2 if you are ready for systems-heavy code.
- If systems feels too steep, first finish Assignment 1 and read the Assignment 2 handout without implementing.

Run:

```bash
source .venv/bin/activate
python labs/week-08-training-systems/bottleneck_map.py
```

What to look for:

- Data parallelism solves throughput by copying the model across workers.
- Tensor parallelism splits large matrix work.
- Pipeline parallelism splits layers.
- FSDP/ZeRO shard parameters, gradients, and optimizer states.
- FlashAttention reduces attention memory traffic.

Questions:

- Which technique reduces memory?
- Which technique increases throughput?
- Which technique adds communication?
- Which technique would help if attention activations are the bottleneck?
