# Study Plan

This plan assumes a Java and agent-development background, with limited Python/PyTorch experience.

## Phase 0: Setup And Python Warmup

Goal: be able to read and run PyTorch training code.

- Create a Python virtual environment.
- Learn enough Python syntax to read assignment code.
- Practice NumPy indexing, broadcasting, and matrix multiplication.
- Practice PyTorch tensors, autograd, modules, optimizers, and dataloaders.

Done when:

- I can train a tiny MLP on synthetic data.
- I can explain tensor shape changes without guessing.

## Phase 1: Tokenization And Data

Goal: understand how text becomes model input.

- Study Unicode, bytes, BPE, vocabulary, and special tokens.
- Implement or trace a small tokenizer.
- Inspect token counts for prompts used in agent work.

Done when:

- I can explain why tokenization affects cost, context length, and model behavior.

## Phase 2: Transformer Core

Goal: understand the forward pass.

- Implement embeddings, attention, MLP, residuals, and normalization.
- Track tensor shapes through a small Transformer.
- Compare a hand-written version to a library implementation.

Done when:

- I can explain one forward pass from token ids to logits.

## Phase 3: Training Loop

Goal: understand pretraining mechanics.

- Build the language-modeling objective.
- Run a tiny training job.
- Track loss, tokens/sec, gradient norms, and samples.
- Learn checkpointing and reproducibility basics.

Done when:

- I can train a toy model and diagnose obvious failures.

## Phase 4: Scaling And Systems

Goal: understand why LLM training is an engineering problem.

- Study batching, GPU memory, mixed precision, and distributed training.
- Learn the vocabulary: data parallelism, tensor parallelism, pipeline parallelism, FSDP.
- Estimate rough memory and compute requirements.

Done when:

- I can make a back-of-the-envelope estimate for a training run.

## Phase 5: Inference And Evaluation

Goal: connect model internals to agent runtime behavior.

- Study sampling, temperature, top-k, top-p, and beam search.
- Study KV cache and prefill/decode.
- Learn basic eval design and failure modes.

Done when:

- I can explain latency and quality tradeoffs in an agent pipeline.

## Phase 6: Alignment Bridge

Goal: connect CS336 to agent training and RLHF.

- Study SFT, reward models, PPO, DPO, and preference data.
- Separate "RL fundamentals" from "LLM alignment recipes."
- Design small preference-data experiments for agent behavior.

Done when:

- I can explain where RLHF depends on RL, and where it is mostly supervised/preference optimization.

