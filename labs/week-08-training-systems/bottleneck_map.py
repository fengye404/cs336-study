from __future__ import annotations


ROWS = [
    (
        "Data parallelism",
        "One full model copy per worker",
        "More examples per step",
        "Gradient all-reduce",
    ),
    (
        "Tensor parallelism",
        "Split large matrix multiplications",
        "Huge layers do not fit or are slow",
        "Frequent activation communication",
    ),
    (
        "Pipeline parallelism",
        "Split model layers across workers",
        "Too many layers for one device",
        "Pipeline bubbles and scheduling",
    ),
    (
        "FSDP / ZeRO",
        "Shard params, grads, optimizer state",
        "Optimizer and model-state memory",
        "Gather/scatter communication",
    ),
    (
        "Activation checkpointing",
        "Recompute activations in backward",
        "Activation memory",
        "Extra compute",
    ),
    (
        "FlashAttention",
        "IO-aware exact attention",
        "Attention memory traffic",
        "Kernel and hardware constraints",
    ),
]


def main() -> None:
    headers = ("Technique", "Core idea", "Helps with", "Cost")
    widths = [28, 36, 34, 34]
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in ROWS:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))

    print()
    print("Rule of thumb")
    print("- Need more throughput: start with data parallelism.")
    print("- Need less optimizer-state memory: look at FSDP/ZeRO.")
    print("- Need less activation memory: use checkpointing.")
    print("- Attention is slow or memory-heavy: understand FlashAttention.")


if __name__ == "__main__":
    main()

