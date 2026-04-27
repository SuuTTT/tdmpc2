# Research Report: TD-MPC2 Codex

## Practical Prior
- TD-MPC2 is sensitive to planning horizon, model size, replay/update ratio, seed, and evaluation episode count.
- For a smoke loop, prioritize runtime-stable interventions over claims of final performance.
- Structural changes should be deferred until the baseline environment, checkpoint creation, and evaluation command are reliable.

## Candidate Levers
- Parameter levers: `steps`, `model_size`, `horizon`, `num_samples`, `seed`, `eval_episodes`.
- Code levers: logging robustness, checkpoint path handling, evaluation wrapper reliability.
- Algorithmic levers: planner refinements and uncertainty handling, only after baseline parity is established.

## Warnings
- Reward shaping inside the environment is not admissible for paper-comparable evaluation.
- Changing evaluation episode count changes variance/cost and must be recorded as a run-tier difference.
