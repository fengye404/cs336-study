from __future__ import annotations

from dataclasses import dataclass


BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
}


@dataclass
class ModelConfig:
    layers: int
    hidden: int
    vocab: int
    seq_len: int
    batch: int
    precision: str


def estimate_params(config: ModelConfig) -> int:
    embedding = config.vocab * config.hidden
    attention = config.layers * 4 * config.hidden * config.hidden
    mlp = config.layers * 8 * config.hidden * config.hidden
    norms = config.layers * 4 * config.hidden
    lm_head = config.vocab * config.hidden
    return embedding + attention + mlp + norms + lm_head


def gb(n_bytes: float) -> float:
    return n_bytes / 1024**3


def report(config: ModelConfig) -> None:
    param_count = estimate_params(config)
    weight_bytes = param_count * BYTES[config.precision]
    grad_bytes = param_count * BYTES[config.precision]
    adam_bytes = param_count * 8

    # Very rough activation estimate: hidden states around each block plus attention-like buffers.
    activation_values = config.batch * config.seq_len * config.hidden * config.layers * 6
    activation_bytes = activation_values * BYTES[config.precision]

    total = weight_bytes + grad_bytes + adam_bytes + activation_bytes

    print("=" * 80)
    print(config)
    print(f"params: {param_count / 1e6:.2f}M")
    print(f"weights: {gb(weight_bytes):.2f} GB")
    print(f"gradients: {gb(grad_bytes):.2f} GB")
    print(f"adam states: {gb(adam_bytes):.2f} GB")
    print(f"activations rough: {gb(activation_bytes):.2f} GB")
    print(f"total rough: {gb(total):.2f} GB")


def main() -> None:
    configs = [
        ModelConfig(layers=2, hidden=64, vocab=128, seq_len=128, batch=32, precision="fp32"),
        ModelConfig(layers=12, hidden=768, vocab=50_000, seq_len=1024, batch=8, precision="bf16"),
        ModelConfig(layers=24, hidden=2048, vocab=50_000, seq_len=2048, batch=4, precision="bf16"),
    ]
    for config in configs:
        report(config)


if __name__ == "__main__":
    main()

