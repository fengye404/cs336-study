# CS336 Study

Personal study repo for Stanford CS336: Language Modeling from Scratch.

This repo is not a mirror of the course. It is a working notebook for learning the LLM training stack from first principles: tokenization, Transformer internals, training loops, data pipelines, scaling, inference, evaluation, and alignment.

## Why This Repo Exists

I am coming from Java and agent development, so the goal is practical:

- Build enough Python and PyTorch fluency to read and modify ML training code.
- Understand what is happening below the agent layer: tokens, attention, loss, sampling, context windows, KV cache, and evaluation.
- Keep a durable record of notes, experiments, mistakes, and implementation details.
- Use CS336 as the spine, then connect it back to agent systems and RLHF later.

> [!NOTE]
> CS336 is a strong LLM systems course, not a general RL course. It is still a great fit before RLHF because it explains the model and training machinery that RLHF sits on top of.

## Official Course Links

- Course site: https://cs336.stanford.edu/
- Course lectures: https://github.com/stanford-cs336/spring2026-lectures
- Course assignments: https://github.com/stanford-cs336
- YouTube: https://www.youtube.com/@stanfordonline

## Repository Map

```text
.
├── assignments/   # Local clones or notes for official assignment repos
├── checkpoints/   # Saved progress snapshots and weekly retrospectives
├── labs/          # Small scratch experiments independent of assignments
├── notes/         # Lecture notes and concept summaries
├── scripts/       # Helper scripts for setup, metrics, or experiments
├── resources.md   # External tutorials and references
└── study-plan.md  # Suggested learning order
```

## Recommended Learning Order

1. Get comfortable with Python, NumPy, and PyTorch tensors.
2. Watch/read CS336 lectures in order, but pause to implement small pieces.
3. Start with tokenizer and Transformer forward pass before worrying about scale.
4. Treat every assignment as a code-reading exercise first, then an implementation task.
5. After pretraining and evaluation, connect the course to SFT, preference learning, RLHF, DPO, and agent behavior.

## Local Setup

Use a dedicated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For official CS336 assignments, prefer each assignment repo's own setup instructions. Course dependencies can change faster than this study repo.

## Study Loop

For each lecture or assignment:

1. Write a short note in `notes/`.
2. Add one small runnable experiment in `labs/` when possible.
3. Record confusion explicitly instead of hiding it.
4. End with a checkpoint: what I can explain, what I can implement, what still feels vague.

Good learning artifacts are not polished essays. They are receipts that thinking happened.

