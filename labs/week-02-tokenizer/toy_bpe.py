from __future__ import annotations

from collections import Counter


Corpus = list[list[str]]


def words_to_symbol_sequences(text: str) -> Corpus:
    return [list(word) + ["</w>"] for word in text.lower().split()]


def count_pairs(corpus: Corpus) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for symbols in corpus:
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += 1
    return counts


def merge_pair(corpus: Corpus, pair: tuple[str, str]) -> Corpus:
    merged = "".join(pair)
    result: Corpus = []

    for symbols in corpus:
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        result.append(new_symbols)

    return result


def train_bpe(text: str, num_merges: int) -> tuple[Corpus, list[tuple[str, str]]]:
    corpus = words_to_symbol_sequences(text)
    merges: list[tuple[str, str]] = []

    print("initial corpus")
    for word in corpus:
        print(word)
    print()

    for step in range(1, num_merges + 1):
        pair_counts = count_pairs(corpus)
        if not pair_counts:
            break

        best_pair, count = pair_counts.most_common(1)[0]
        corpus = merge_pair(corpus, best_pair)
        merges.append(best_pair)

        print(f"merge {step}: {best_pair} count={count}")
        for word in corpus:
            print(word)
        print()

    return corpus, merges


def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols = list(word.lower()) + ["</w>"]
    for pair in merges:
        symbols = merge_pair([symbols], pair)[0]
    return symbols


def main() -> None:
    text = "low lower lowest new newer newest low lower"
    _, merges = train_bpe(text, num_merges=8)

    print("learned merges")
    for merge in merges:
        print(merge)
    print()

    for word in ["low", "lowest", "newer", "unknown"]:
        print(f"{word:>8} -> {encode_word(word, merges)}")


if __name__ == "__main__":
    main()

