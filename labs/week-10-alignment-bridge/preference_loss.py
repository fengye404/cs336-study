from __future__ import annotations

import torch
import torch.nn.functional as F


EXAMPLES = [
    {
        "prompt": "User asks for a safe shell command",
        "chosen": "Explain the command, then provide the smallest safe command.",
        "rejected": "Run a broad destructive command without explanation.",
    },
    {
        "prompt": "User asks an ambiguous agent task",
        "chosen": "Ask one clarifying question before taking action.",
        "rejected": "Invent missing requirements and proceed.",
    },
]


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    policy_log_ratio = policy_chosen_logp - policy_rejected_logp
    ref_log_ratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_log_ratio - ref_log_ratio)
    return -F.logsigmoid(logits).mean()


def main() -> None:
    for i, example in enumerate(EXAMPLES, start=1):
        print("=" * 80)
        print(f"example {i}")
        print(f"prompt:   {example['prompt']}")
        print(f"chosen:   {example['chosen']}")
        print(f"rejected: {example['rejected']}")

    policy_chosen = torch.tensor([-12.0, -8.0])
    policy_rejected = torch.tensor([-15.0, -7.5])
    ref_chosen = torch.tensor([-11.5, -8.5])
    ref_rejected = torch.tensor([-13.0, -8.0])

    loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    print()
    print("toy DPO numbers")
    print(f"policy chosen logp:   {policy_chosen.tolist()}")
    print(f"policy rejected logp: {policy_rejected.tolist()}")
    print(f"ref chosen logp:      {ref_chosen.tolist()}")
    print(f"ref rejected logp:    {ref_rejected.tolist()}")
    print(f"dpo loss: {loss.item():.4f}")
    print()
    print("Interpretation:")
    print("Lower loss means the policy prefers chosen over rejected more strongly than the reference does.")


if __name__ == "__main__":
    main()

