# Research Log: TD-MPC2 Codex AutoSOTA Run

## 2026-04-27: Repository and Environment Setup

- Preserved the prior edited TD-MPC2 workspace by renaming `/workspace/tdmpc2` to `/workspace/tdmpc2-copilot`.
- Fresh-cloned upstream TD-MPC2 into `/workspace/tdmpc2-codex`.
- Tagged the clean upstream state as `_baseline` at commit `8bbc14e`.
- Added AutoSOTA control artifacts:
  - `autosota.yaml`
  - `objective.md`
  - `red_lines.md`
  - `code_analysis.md`
  - `research_report.md`
  - `idea_library.md`
  - `scores.jsonl`
- Built the `tdmpc2` conda environment from `docker/environment.yaml`.
- Encountered disk exhaustion while installing `dm-control`.
- Repaired by cleaning conda and pip caches, then resumed pip installation with `--no-cache-dir`.
- Verified core imports and CUDA availability.

## 2026-04-27: Baseline Smoke Tier

- Ran a short scheduler-validation baseline:
  - task: `dog-run`
  - model size: `5`
  - train steps: `1000`
  - seed: `1`
  - eval episodes: `1`
- Result:
  - reward: `6.3`
  - success: `0.0`
- Recorded as iteration 0 in `scores.jsonl`.
- Tagged `_best` to the baseline smoke-result commit.

## 2026-04-27: Iteration 1

- Tested a small training-budget increase:
  - idea: `IDEA-002`
  - task: `dog-run`
  - model size: `5`
  - train steps: `2000`
  - seed: `1`
  - eval episodes: `1`
- Result:
  - reward: `6.3`
  - decision: `NO_IMPROVEMENT`
- `_best` remained unchanged.

## 2026-04-27: Human Prior Ingest

- Added a new plugin skill:
  - `/workspace/autosota-lite/plugins/autosota-lite/skills/autosota-human-idea-ingest/SKILL.md`
- Captured the user prior:
  - abstraction may enhance TD-MPC2.
- Updated `research_report.md` with the abstraction hypothesis and related literature direction.
- Updated `idea_library.md` with:
  - `IDEA-004`: latent bottleneck abstraction probe.
  - `IDEA-005`: discrete latent codebook abstraction, review required.
  - `IDEA-006`: temporal abstraction planner, review required.

## 2026-04-27: Iteration 2 Abstraction Probe

- Ran a background abstraction probe:
  - idea: `IDEA-004`
  - task: `dog-run`
  - model size: `1`
  - train steps: `50000`
  - seed: `1`
  - eval episodes: `1`
- Mechanism:
  - `model_size=1` reduces TD-MPC2 latent dimension from the baseline model-size-5 latent dimension of `512` to `128`.
  - This is a low-risk proxy for latent bottleneck abstraction because it uses TD-MPC2's existing configuration rather than changing reward, task, or evaluation code.
- Result:
  - reward: `55.2`
  - success: `0.0`
  - decision in `scores.jsonl`: `IMPROVED`

## Important Validity Note

The iteration 2 abstraction result is only improved relative to the earlier short smoke baseline (`model_size=5`, `1000` steps, reward `6.3`) and iteration 1 (`model_size=5`, `2000` steps, reward `6.3`).

That is not a compute-matched scientific comparison. The abstraction run used `50000` training steps, while the previous default-model baselines used `1000` and `2000` steps.

Therefore, the current claim should be stated as:

> The `model_size=1` abstraction/bottleneck run produced a much higher smoke-tier reward than the earlier short scheduler-validation baselines.

It should not yet be stated as:

> Abstraction outperforms original TD-MPC2.

The fair next comparison is to run the original/default TD-MPC2 configuration with `model_size=5`, `50000` training steps, same task, same seed, same evaluation command tier, and compare that result against the `model_size=1` abstraction result.

## Next Action

Run a compute-matched comparator:

- task: `dog-run`
- model size: `5`
- train steps: `50000`
- seed: `1`
- eval episodes: `1`
- experiment name: `iter3_original_m5_steps50000`

Interpretation:

- If `model_size=5, steps=50000` beats `55.2`, then the abstraction/bottleneck hypothesis is not supported under this smoke tier.
- If `model_size=1, steps=50000` remains competitive or better, then the human abstraction prior has real evidence and should be followed by stronger evaluation, preferably `eval_episodes=5` or `10`.

## 2026-04-28: Validation and Speed Follow-Up

- The 10-episode validation completed:
  - `model_size=1`, `50000` steps: reward `50.5`.
  - `model_size=5`, `50000` steps: reward `29.3`.
- The abstraction/bottleneck probe remains ahead under the matched 50k-step validation tier.
- Runtime remains the bottleneck: 50k steps takes roughly 1.5 hours on the current RTX 3060 instance.
- Next scheduled idea is `IDEA-007`, a planner-cost reduction probe:
  - `model_size=1`
  - `steps=50000`
  - `num_samples=256`
  - `num_elites=32`
  - `num_pi_trajs=12`
  - `iterations=4`
  - `eval_episodes=10`
- Success criterion:
  - materially lower wall-clock runtime than the default `model_size=1` 50k run.
  - reward remains competitive with the validated abstraction score of `50.5`.
