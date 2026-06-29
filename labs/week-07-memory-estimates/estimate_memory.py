from __future__ import annotations

from dataclasses import dataclass


BYTES = {
    # 不同精度的每个数值占用字节数。bf16/fp16 是 fp32 的一半。
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
    # 这是粗略估算，不追求和某个具体架构完全一致。
    # 目的是建立“参数量来自哪里”的直觉。
    embedding = config.vocab * config.hidden
    # 一个 Transformer 层里，attention 大致有 q/k/v/out 四个 hidden x hidden 矩阵。
    attention = config.layers * 4 * config.hidden * config.hidden
    # MLP 常见扩展倍率是 4x：hidden -> 4hidden -> hidden，所以约 8 * hidden^2。
    mlp = config.layers * 8 * config.hidden * config.hidden
    norms = config.layers * 4 * config.hidden
    lm_head = config.vocab * config.hidden
    return embedding + attention + mlp + norms + lm_head


def gb(n_bytes: float) -> float:
    return n_bytes / 1024**3


def report(config: ModelConfig) -> None:
    param_count = estimate_params(config)
    # 训练时不只存 weights，还要存 gradients 和 optimizer states。
    weight_bytes = param_count * BYTES[config.precision]
    grad_bytes = param_count * BYTES[config.precision]
    # AdamW 通常为每个参数维护两个 fp32 状态：m 和 v，所以约 8 bytes/param。
    adam_bytes = param_count * 8

    # 非常粗略的 activation 估算：每层会保留若干中间 hidden states 和 attention 相关 buffer。
    # 真实显存还受实现、checkpointing、kernel 等影响。
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
