# Code Cognition Map: TD-MPC2 Codex

## Objective and Metric
- Target task: `dog-run`.
- Metric: episode reward from `tdmpc2/evaluate.py`.
- Baseline command tier: short smoke run first, full paper run later.

## Entrypoints
- Training: `tdmpc2/train.py`.
- Evaluation: `tdmpc2/evaluate.py`.
- Config: `tdmpc2/config.yaml`.
- Environment factory: `tdmpc2/envs/__init__.py`.

## Pipeline
- Hydra parses `tdmpc2/config.yaml`.
- `train.py` builds an environment, `TDMPC2` agent, replay buffer, and trainer.
- `evaluate.py` loads a checkpoint, creates the task environment, and reports episode reward.

## Constraints
- Evaluation semantics are frozen for a given run tier.
- Reward/task definitions are frozen.
- Any smoke-run result must be labeled as smoke evidence, not paper reproduction.

## Retrieval Pointers
- Search config knobs in `tdmpc2/config.yaml`.
- Search agent planning logic in `tdmpc2/tdmpc2.py`.
- Search trainer behavior under `tdmpc2/trainer/`.
