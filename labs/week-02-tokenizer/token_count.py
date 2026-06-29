from __future__ import annotations

import tiktoken


PROMPTS = [
    "Summarize this document in three bullets.",
    "请把这篇文章总结成三条要点。",
    "Use the browser tool, search for CS336, then extract the assignment links.",
    "forward -> loss -> backward -> optimizer.step",
    "hello world",
    "hello     world",
]


def main() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")

    for prompt in PROMPTS:
        token_ids = encoding.encode(prompt)
        print("=" * 80)
        print(prompt)
        print(f"chars: {len(prompt)}")
        print(f"tokens: {len(token_ids)}")
        print(f"token ids: {token_ids}")
        print(f"decoded pieces: {[encoding.decode([token_id]) for token_id in token_ids]}")


if __name__ == "__main__":
    main()

