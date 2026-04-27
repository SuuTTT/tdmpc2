# Objective: TD-MPC2 Codex Baseline

## Target Claim
Reproduce and improve a small, scheduler-managed TD-MPC2 run on DMControl `dog-run`.

## Metric
- Primary metric: episode reward.
- Direction: higher is better.
- Baseline source: measured inside this Vast.ai container.

## Initial Scope
- Use short smoke training/evaluation first to validate the Scheduler loop.
- Do not claim paper-level reproduction from smoke runs.
- Scale to the full paper protocol only after the environment and lifecycle loop are stable.

## Acceptance Criteria
- Clean baseline git state tagged `_baseline`.
- Baseline result recorded in `scores.jsonl`.
- Each optimization iteration records `PRE_COMMIT`, command, score, and decision.
