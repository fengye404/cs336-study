# Week 08: Efficient Training And Distributed Concepts

Goal: map training-system techniques to the bottlenecks they address.

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

