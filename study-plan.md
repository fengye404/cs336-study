# CS336 Fast Track Study Plan

This is the faster plan.

It adapts Stanford CS336: Language Modeling from Scratch to a working schedule:

- Work starts at 10:00.
- Work ends around 22:00.
- Weekdays are short but consistent.
- Weekends carry the real implementation load.

Target duration: 10 weeks.

Expected load: 12-15 hours/week.

This is close to the official CS336 course tempo, but with a working-person schedule. The tradeoff is simple: fewer detours, less perfectionism, more code.

## Weekly Rhythm

| Time | Task | Target |
| --- | --- | --- |
| Monday 08:15-09:30 | Watch/read new CS336 material | Concept intake |
| Tuesday 08:30-09:15 | Python/PyTorch drill tied to the topic | Skill warmup |
| Wednesday 08:15-09:30 | Continue lecture/notes | Concept intake |
| Thursday 08:30-09:15 | Small implementation step | Code momentum |
| Friday 22:30-23:00 | List blockers and plan weekend | Planning |
| Saturday 09:30-12:30 | Main assignment/lab block | Deep code |
| Saturday 14:30-17:30 | Continue implementation | Deep code |
| Sunday 10:00-13:00 | Debug, test, clean up | Finish |
| Sunday 20:30-21:15 | Write checkpoint | Retention |

Minimum viable week:

- Watch/read 90 minutes total.
- Write one note.
- Make one commit.
- Record blockers in `checkpoints/`.

## Speed Rules

- Do not wait until Python feels comfortable. Learn Python through the assignment.
- Do not rewrite official code for beauty.
- Do not read three tutorials for one topic. Use CS336 first, then one backup resource only when blocked.
- Every week ends with a concrete artifact: code, notes, experiment log, or checkpoint.
- If a concept is unclear but not blocking code, keep moving and mark it in `notes/confusions.md`.

## 10-Week Plan

### Week 1: Bootstrapping Python, PyTorch, And CS336

Goal: get the environment working and build enough PyTorch muscle memory to start assignments.

Tasks:

- Set up `.venv` and install dependencies.
- Skim the CS336 course site and current assignment structure.
- Learn Python essentials: functions, classes, list/dict comprehensions, iterators, file I/O.
- Learn PyTorch essentials: tensors, autograd, `nn.Module`, optimizers.
- Lab: train a tiny MLP on synthetic data.

Deliverables:

- `labs/week-01-pytorch-basics/`
- `notes/week-01-setup.md`
- `checkpoints/week-01.md`

Done when:

- I can run a PyTorch training loop and explain what forward, loss, backward, and optimizer step do.

### Week 2: Tokenization And BPE

Goal: understand how text becomes model input.

Tasks:

- Study Unicode, UTF-8, bytes, vocabulary, merges, and special tokens.
- Start the CS336 tokenizer assignment or equivalent local lab.
- Compare token counts for real agent prompts.
- Implement or trace a small BPE tokenizer.

Deliverables:

- `labs/week-02-tokenizer/`
- `notes/week-02-tokenization.md`
- Token-count examples from agent prompts.

Done when:

- I can explain training a tokenizer versus using a trained tokenizer for inference.

### Week 3: Embeddings, Logits, And The LM Objective

Goal: understand the simplest next-token prediction model.

Tasks:

- Study token embeddings, positional information, logits, softmax, and cross entropy.
- Build a minimal next-token predictor.
- Print tensor shapes through the full path.
- Read the relevant CS336 material before coding deeper.

Deliverables:

- `labs/week-03-lm-objective/`
- `notes/week-03-language-modeling.md`

Done when:

- I can explain input token ids to logits to loss for one batch.

### Week 4: Attention

Goal: understand causal self-attention well enough to implement it.

Tasks:

- Study `Q`, `K`, `V`, scaled dot-product attention, causal masks, and multi-head attention.
- Implement attention from scratch.
- Add shape assertions or comments.
- Test masking behavior with a tiny example.

Deliverables:

- `labs/week-04-attention/`
- `notes/week-04-attention.md`

Done when:

- I can trace attention shapes without guessing.

### Week 5: Transformer Block To Tiny GPT

Goal: assemble the model core.

Tasks:

- Study residual connections, normalization, MLP, activations, and dropout.
- Implement one Transformer block.
- Stack blocks into a tiny GPT-style model.
- Generate samples, even if they are bad.

Deliverables:

- `labs/week-05-tiny-gpt/`
- `notes/week-05-transformer-block.md`

Done when:

- I can explain a full forward pass from text to next-token logits.

### Week 6: Training Loop And Debugging

Goal: make training runs reliable enough to learn from.

Tasks:

- Study batching, gradient accumulation, clipping, schedulers, and checkpointing.
- Add validation loss.
- Add checkpoint save/load.
- Run at least three small experiments changing one variable at a time.

Deliverables:

- `labs/week-06-training-loop/`
- `checkpoints/week-06-experiment-log.md`

Done when:

- I can stop/resume training and compare two runs using loss curves.

### Week 7: Scaling Basics And GPU Memory

Goal: understand why LLM training becomes a systems problem.

Tasks:

- Study parameters, activations, optimizer states, batch size, and sequence length.
- Estimate memory for toy model sizes.
- Study fp32, fp16, bf16, and numerical stability.
- If hardware allows, compare mixed precision versus fp32.

Deliverables:

- `notes/week-07-scaling-memory.md`
- `labs/week-07-memory-estimates/`

Done when:

- I can estimate whether a model/run is likely to fit before launching it.

### Week 8: Efficient Training And Distributed Concepts

Goal: learn the vocabulary of modern training systems.

Tasks:

- Study FlashAttention at the concept level.
- Study data parallelism, tensor parallelism, pipeline parallelism, and FSDP.
- Read one CS336 systems lecture carefully.
- Write a map of which technique solves which bottleneck.

Deliverables:

- `notes/week-08-training-systems.md`

Done when:

- I can read a distributed-training discussion without getting lost in names.

### Week 9: Inference, Sampling, And Evaluation

Goal: connect model internals to agent runtime behavior.

Tasks:

- Study greedy decoding, temperature, top-k, top-p, prefill, decode, and KV cache.
- Compare generated outputs under sampling settings.
- Study perplexity, benchmark design, contamination, and qualitative evals.
- Create a tiny eval set for agent-style prompts.

Deliverables:

- `labs/week-09-inference-eval/`
- `notes/week-09-kv-cache-eval.md`

Done when:

- I can explain latency and quality tradeoffs in an agent pipeline.

### Week 10: Alignment Bridge: SFT, Preferences, RLHF, DPO

Goal: connect CS336 back to agent development and RL.

Tasks:

- Study SFT, instruction data, preference pairs, reward models, PPO, and DPO.
- Map agent traces to possible SFT data and preference data.
- Write a final summary: what I now understand about LLM training and what to learn next.

Deliverables:

- `notes/week-10-alignment-bridge.md`
- `checkpoints/week-10-final-review.md`

Done when:

- I can explain where RLHF depends on RL fundamentals and where preference optimization is enough.

## What To Skip On The Fast Track

Skip these unless they are directly blocking the current assignment:

- Deep math proofs.
- Building a polished library structure.
- Full distributed training implementation.
- Reading every optional paper.
- Optimizing tiny labs for performance.
- Perfect notes.

## First Two Weeks Checklist

Week 1:

- [ ] Environment works.
- [ ] One PyTorch training loop runs.
- [ ] One checkpoint file exists.

Week 2:

- [ ] Tokenization notes exist.
- [ ] A small tokenizer lab exists.
- [ ] I can explain BPE without looking at notes.

If these are done, the plan is moving fast enough.
