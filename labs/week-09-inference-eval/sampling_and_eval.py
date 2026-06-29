from __future__ import annotations

import math

import torch
import torch.nn.functional as F


VOCAB = ["plan", "search", "read", "write", "ask", "stop"]


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    # top-k 只保留分数最高的 k 个 token，其它 token 设成 -inf。
    # softmax 后，-inf 对应概率 0。
    values, _ = torch.topk(logits, k)
    cutoff = values[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    # top-p 也叫 nucleus sampling：保留累计概率达到 p 的最小 token 集合。
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > p
    # 至少保留概率最高的 token，避免候选集为空。
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # scatter 把排序后的 logits 放回原始 vocab 顺序。
    result = torch.full_like(logits, float("-inf"))
    result.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return result


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None):
    # temperature < 1 会让分布更尖锐；temperature > 1 会让分布更平。
    logits = logits / temperature
    if top_k is not None:
        logits = apply_top_k(logits, top_k)
    if top_p is not None:
        logits = apply_top_p(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    # 按概率采样。生成模型不一定总选概率最高的 token。
    idx = torch.multinomial(probs, num_samples=1).item()
    return VOCAB[idx], probs


def nll_and_perplexity(log_probs: torch.Tensor, target_ids: torch.Tensor) -> tuple[float, float]:
    # NLL: 真实 token 的负 log probability，越低越好。
    nll = -log_probs[torch.arange(len(target_ids)), target_ids].mean().item()
    # perplexity = exp(NLL)。可以粗略理解成“模型平均有多困惑”。
    return nll, math.exp(nll)


def main() -> None:
    torch.manual_seed(5)
    # 这里手写一组 logits，模拟模型对 6 个 action token 的偏好。
    logits = torch.tensor([2.2, 1.7, 1.2, 0.9, 0.4, -0.5])

    print("base probabilities")
    base_probs = F.softmax(logits, dim=-1)
    for token, prob in zip(VOCAB, base_probs.tolist()):
        print(f"{token:>6}: {prob:.3f}")
    print()

    print(f"greedy: {VOCAB[torch.argmax(logits).item()]}")
    for label, kwargs in [
        ("temperature=0.5", {"temperature": 0.5}),
        ("temperature=1.5", {"temperature": 1.5}),
        ("top_k=3", {"top_k": 3}),
        ("top_p=0.75", {"top_p": 0.75}),
    ]:
        token, probs = sample(logits, **kwargs)
        visible = [(tok, round(prob, 3)) for tok, prob in zip(VOCAB, probs.tolist()) if prob > 0]
        print(f"{label:>16} -> sampled={token}, candidates={visible}")

    print()
    print("tiny eval")
    # fake_log_probs 模拟三步预测，每一行是一个位置的 log-probabilities。
    fake_log_probs = torch.log_softmax(
        torch.tensor(
            [
                [3.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                [0.0, 2.5, 0.5, 0.0, 0.0, -1.0],
                [0.0, 0.0, 2.2, 1.5, 0.0, -1.0],
            ]
        ),
        dim=-1,
    )
    # target_ids 表示每一步真实应该选哪个 token。
    target_ids = torch.tensor([0, 1, 2])
    nll, ppl = nll_and_perplexity(fake_log_probs, target_ids)
    print(f"nll={nll:.4f}")
    print(f"perplexity={ppl:.4f}")


if __name__ == "__main__":
    main()
