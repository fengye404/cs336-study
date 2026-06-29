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
    # cl100k_base 是 OpenAI 常用 tokenizer 之一。
    # 这里用它观察“同一句话到底会被切成多少 token”。
    encoding = tiktoken.get_encoding("cl100k_base")

    for prompt in PROMPTS:
        # encode: 文本 -> token ids。模型实际看到的是这些整数 id。
        token_ids = encoding.encode(prompt)
        print("=" * 80)
        print(prompt)
        print(f"chars: {len(prompt)}")
        print(f"tokens: {len(token_ids)}")
        print(f"token ids: {token_ids}")
        # 把每个 token 单独 decode 回来，帮助你看到 tokenizer 的切分边界。
        print(f"decoded pieces: {[encoding.decode([token_id]) for token_id in token_ids]}")


if __name__ == "__main__":
    main()
