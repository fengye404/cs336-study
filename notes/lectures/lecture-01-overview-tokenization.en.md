# Lecture 01: Overview And Tokenization

Source: `external/cs336-lectures/lecture_01.py`

This is the English text companion for Lecture 01, condensed from the official executable lecture.

If you only use one study artifact, use the executable notebook:

- `notes/lectures/lecture-01-overview-tokenization.zh.ipynb`

This Markdown file is best for quick review and terminology lookup. The notebook follows the same lecture rhythm and adds runnable experiments at the matching sections.

## 1. Why This Course Exists

Researchers and builders have moved up layers of abstraction:

- In 2016, many researchers implemented and trained their own models.
- Around 2018, many downloaded pretrained models such as BERT and fine-tuned them.
- Today, many people prompt API models such as GPT, Claude, and Gemini.

This abstraction is productive, but language-model abstractions are leaky. To do fundamental work, you need to understand the stack below the API: tokenization, model architecture, training, systems, data, evaluation, and alignment.

The course philosophy is: learn by building.

## 2. Frontier Models Are Out Of Reach

Frontier models are too expensive for a class or personal study repo to reproduce. The important question is not "can we train GPT-4 from scratch?" The useful question is:

> What knowledge transfers from small, buildable models to frontier-scale models?

The lecture separates transferable knowledge into three categories:

- Mechanics: how Transformers, tokenizers, optimizers, parallelism, and training loops work.
- Mindset: treat compute, memory, bandwidth, and data as scarce resources.
- Intuitions: empirical design choices about data and modeling. These transfer less reliably across scale.

## 3. The Main Framing: Efficiency

The course repeatedly returns to one framing:

> Build the best model you can under fixed compute, memory, communication, and data constraints.

The bitter lesson is not "scale is all that matters." A better interpretation is:

> Algorithms that scale are what matter.

This matters because waste becomes more expensive at larger scale. Good model architecture, good kernels, good data filtering, good batching, and good hyperparameter choices are all forms of efficiency.

## 4. A Short History Of Language Models

Pre-neural era:

- Shannon studied language modeling as a way to measure the entropy of English.
- N-gram models were used in machine translation and speech recognition.

Neural era:

- LSTMs and early neural language models introduced learned sequence modeling.
- Sequence-to-sequence models helped machine translation.
- Attention and Transformers became the foundation for modern LMs.
- Optimizers such as Adam made deep model training more practical.

Scaling era:

- GPT-2 showed fluent generation and early zero-shot behavior.
- Scaling laws made model performance more predictable.
- GPT-3 showed in-context learning at large scale.
- Chinchilla-style results emphasized compute-optimal tradeoffs between model size and data size.

Open-model era:

- Open weights and open-source training efforts make it possible to study modern LM ideas.
- CS336 uses these public ideas to teach the underlying stack.

## 5. What Is A Language Model?

A language model places a probability distribution over token sequences.

Historically, the interface changed:

- 2018: a model you fine-tune.
- 2020: a model you prompt.
- 2022: a model you chat with.
- 2026: a model that can act as an agent.

The fundamentals are still shared: tokens, attention, optimization, kernels, data, and evaluation.

## 6. Course Syllabus

Assignment 1: Basics

- Tokenization
- Transformer architecture
- Cross-entropy loss
- AdamW optimizer
- Training loop
- Resource accounting

Assignment 2: Systems

- Kernels
- GPU efficiency
- Parallelism
- Inference

Assignment 3: Scaling Laws

- Predict how performance changes with model size, data, and compute.

Assignment 4: Data

- Evaluation
- Curation
- Filtering
- Deduplication
- Data mixing

Assignment 5: Alignment

- Supervised fine-tuning
- Preference learning
- RLHF-related algorithms and systems

## 7. Tokenization: The First Unit

Notebook pointer: start at section 7 and run the shared helper cell.

Tokenization asks:

> What are the atoms that the model operates on?

Formally, a tokenizer converts between raw input text and integer token sequences:

- Encode: string -> list of token ids
- Decode: list of token ids -> string

Language models usually do not directly operate on Python strings. They operate on integer ids, which are then mapped into vectors by an embedding table.

## 8. Character Tokenization

Notebook pointer: section 8, run the `CharacterTokenizer` cell.

A Unicode string can be treated as a sequence of characters. Each character has a code point:

- `ord("a") == 97`
- `chr(97) == "a"`

Pros:

- Simple.
- Can round-trip text.

Cons:

- Unicode has many possible characters.
- Rare characters waste vocabulary capacity.
- Compression is poor compared with modern tokenizers.

## 9. Byte Tokenization

Notebook pointer: section 9, run the `ByteTokenizer` cell and compare UTF-8 bytes for Chinese text and emoji.

A Unicode string can also be encoded as UTF-8 bytes. Each byte is an integer from 0 to 255.

Pros:

- Small fixed vocabulary: 256 byte values.
- Can represent any UTF-8 text.

Cons:

- Poor compression: one token per byte.
- Sequences become long.
- Long sequences are costly because standard Transformer attention scales quadratically with sequence length.

## 10. Word Tokenization

Notebook pointer: section 10, run the toy word tokenizer and inspect long words and Chinese text.

Classic NLP often split text into words or word-like chunks.

Pros:

- Tokens are human-meaningful.
- Compression can be good.

Cons:

- Vocabulary can become huge.
- Rare words are hard to learn.
- Unseen words require an unknown-token strategy.
- Fixed vocabulary size is awkward.

## 11. Byte Pair Encoding

Notebook pointer: sections 13-15, run `merge`, `train_tiny_bpe`, and `TinyBPETokenizer` in order.

Byte Pair Encoding, or BPE, is a compromise between bytes and words.

Basic idea:

1. Start with bytes as the initial tokens.
2. Count adjacent token pairs in training text.
3. Merge the most frequent adjacent pair into a new token.
4. Repeat until the vocabulary reaches the desired size.

Intuition:

- Common byte sequences become single tokens.
- Rare sequences remain decomposed into smaller pieces.
- The tokenizer learns a vocabulary adapted to the data.

This gives a fixed vocabulary while usually producing much shorter sequences than byte-level tokenization.

## 12. Vocabulary Size And Compression

Notebook pointer: sections 11-12, inspect a real tokenizer with `tiktoken`, then compare compression ratios.

Two pressures are in tension:

- Bigger vocabulary can improve compression because common chunks become single tokens.
- Bigger vocabulary also makes the model's embedding and output layers larger and sparser.

Compression ratio in the lecture means:

```text
number of UTF-8 bytes / number of tokens
```

A higher ratio means fewer tokens for the same text, which usually helps attention cost.

## 13. Assignment 1 Preview

In Assignment 1, you will implement:

- A BPE tokenizer
- A Transformer
- Cross-entropy loss
- AdamW
- A training loop
- Resource accounting

The high-level balancing act:

- Expressivity: can the model represent complex dependencies?
- Stability: do activations and gradients stay in a usable range?
- Efficiency: does it train and run fast on real hardware?

## 14. How This Connects To Week 1

Notebook pointer: sections 17-18, use the final checkpoint questions to verify today’s study block.

Your Week 1 tiny MLP is not a language model, but it teaches the training loop that later appears inside language modeling:

```text
input -> model -> prediction -> loss -> backward -> optimizer step
```

In a language model, the same structure becomes:

```text
token ids -> Transformer -> logits -> cross-entropy loss -> backward -> optimizer step
```

For Week 1, your goal is not to master tokenization yet. Your goal is to understand why CS336 starts with tokenization and why every later piece depends on the training loop.
