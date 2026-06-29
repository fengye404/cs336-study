# Week 10: Alignment Bridge

Goal: connect LLM training to SFT, preferences, RLHF, and DPO.

## Where This Fits

Lecture connection:

- Mid/post-training, SFT, RLHF/RLVR, and alignment.

Official assignment connection:

- Assignment 5: Alignment and reasoning RL.

Before this lab:

- Finish Lab 09.
- Know what log-probabilities are from the language-modeling objective.

After this lab:

- Start Assignment 5 only after you can explain SFT data versus preference data.
- If PPO/RL feels vague, pause and study RL basics separately before going deep on RLHF.

Run:

```bash
source .venv/bin/activate
python labs/week-10-alignment-bridge/preference_loss.py
```

What to look for:

- Preference data compares a chosen answer with a rejected answer.
- DPO uses policy log-probabilities and reference log-probabilities.
- The loss rewards the policy for preferring chosen responses more than the reference does.

Questions:

- What data would an agent trace provide for SFT?
- What data would require human or model preference labels?
- Why is DPO not the same thing as general RL?
