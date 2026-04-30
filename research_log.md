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

## 2026-04-28: Iteration 5 Fast Planner Result

- Completed the planner-cost reduction probe:
  - idea: `IDEA-007`
  - task: `dog-run`
  - model size: `1`
  - train steps: `50000`
  - seed: `1`
  - eval episodes: `10`
  - planner settings:
    - `num_samples=256`
    - `num_elites=32`
    - `num_pi_trajs=12`
    - `iterations=4`
- Result:
  - reward: `47.8`
  - success: `0.0`
  - train time: `1:18:17`
  - decision: `FASTPLANNER_COMPETITIVE`
- Interpretation:
  - The fast planner is slightly below the validated default abstraction score (`47.8` vs `50.5`).
  - The result is still competitive under the 50k-step, 10-episode tier.
  - Runtime improved relative to the earlier rough default `model_size=1` 50k runtime estimate of about 1.5 hours.
- Evidence recorded in:
  - `scores.jsonl`
  - `.autosota_scheduler_test/iter5_fastplanner_state.json`
  - `logs/autosota/iter5_fastplanner_m1_steps50000.log`
  - `logs/autosota/iter5_fastplanner_m1_steps50000_eval10.log`

## Current State

- Git worktree is clean.
- Best reward under the validated 50k-step, 10-episode tier remains:
  - `model_size=1`, default planner: reward `50.5`.
- Best speed/performance tradeoff currently observed:
  - `model_size=1`, reduced planner: reward `47.8`, train time `1:18:17`.
- Main finding so far:
  - The human abstraction prior is supported in this smoke-tier setup: the `model_size=1` latent bottleneck outperformed the compute-matched `model_size=5` comparator on `dog-run`.
- Important limitation:
  - These are still smoke-tier runs on one seed and short training budgets, not paper-level reproduction.

## Recommended Next Action

Run a focused follow-up on the abstraction path rather than moving to high-risk code changes:

- Option A: repeat the validated winner with another seed:
  - `model_size=1`
  - default planner
  - `steps=50000`
  - `eval_episodes=10`
  - new seed, for example `seed=2`
- Option B: tune between default and fast planner:
  - keep `model_size=1`
  - test a milder planner reduction than iteration 5, such as `num_samples=384`, `num_elites=48`, `num_pi_trajs=16`, `iterations=4`
  - goal: recover reward closer to `50.5` while preserving most of the runtime gain.
- Defer `IDEA-005` and `IDEA-006` until the low-risk parameter evidence is stronger across seeds.

## 2026-04-28: Iteration 6 Acrobot Compile Run Launched

- Launched a cross-task abstraction probe on:
  - task: `acrobot-swingup`
  - idea: `IDEA-004`
  - model size: `1`
  - train steps: `4000000`
  - seed: `1`
  - eval frequency: `100000`
  - eval episodes: `10`
  - `compile=true`
- Comparison target:
  - official local CSV: `results/tdmpc2/acrobot-swingup.csv`
  - official schema: `step,reward,seed`
  - official seeds: `1`, `2`, `3`
  - official final step: `4000000`
- Output paths:
  - run script: `scripts/autosota_iter6_acrobot_compile_abstraction.sh`
  - exporter: `scripts/export_acrobot_compile_result.py`
  - train log: `logs/autosota/iter6_acrobot_compile_abstraction_m1_steps4000000.log`
  - official-format candidate CSV: `results/tdmpc2-codex/acrobot-swingup_abstraction_compile.csv`
  - comparison CSV: `results/tdmpc2-codex/acrobot-swingup_abstraction_compile_compare.csv`
- Early compile status:
  - a 1-step compile smoke test completed successfully.
  - the smoke test took about `1:20`, indicating high first-compile overhead.
  - the full run started and wrote the initial step-0 evaluation: reward `8.6`.
  - PyTorch emitted recompilation warnings during startup/pretraining, so the acceleration claim should wait for post-warmup throughput.
- In-progress milestone:
  - step `100000`: candidate reward `112.0`; official TD-MPC2 mean at same step `179.1`.
  - step `200000`: candidate reward `285.7`; official TD-MPC2 mean at same step `296.3`.
  - step `300000`: candidate reward `447.8`; official TD-MPC2 mean at same step `346.9`.
  - step `400000`: candidate reward `497.6`; official TD-MPC2 mean at same step `322.0`.
  - elapsed time at step `400000`: `2:06:09`.
- Current interpretation:
  - This run is designed to test whether the dog-run abstraction result transfers to a different DMControl task while also testing PyTorch compile.
  - It is not a like-for-like model-size comparison with official TD-MPC2, because the official single-task result is model-size 5 and the abstraction candidate intentionally uses model-size 1.

## 2026-04-28: Iteration 7 Information Bottleneck Probe Prepared

- Added `IDEA-008`, an information-theoretic abstraction method.
- Mechanism:
  - TD-MPC2 already represents state with grouped SimNorm latent probabilities.
  - The new optional loss computes KL from each SimNorm group to a uniform prior.
  - This is an information-bottleneck proxy: it discourages the latent from carrying extra observation-specific information unless the main TD-MPC2 losses need it.
- Code changes:
  - `tdmpc2/config.yaml`: added `ib_coef`, default `0.0`.
  - `tdmpc2/tdmpc2.py`: added `_latent_ib_loss` and logs `ib_loss`.
  - Default behavior is unchanged when `ib_coef=0.0`.
- Prepared run:
  - script: `scripts/autosota_iter7_acrobot_ib_abstraction.sh`
  - task: `acrobot-swingup`
  - model size: `1`
  - steps: `400000`
  - seed: `1`
  - eval frequency: `100000`
  - eval episodes: `10`
  - `ib_coef=0.01`
  - `compile=false`
- The run script waits for the active iteration 6 compile run before starting, to avoid GPU contention.
- Planned outputs:
  - official-format CSV: `results/tdmpc2-codex/acrobot-swingup_ib_abstraction.csv`
  - comparison CSV: `results/tdmpc2-codex/acrobot-swingup_ib_abstraction_compare.csv`

## 2026-04-28: Iteration 8 Structural Entropy Probe Prepared

- Used the requested skill:
  - `/workspace/autosota-lite/.codex/skills/structural-entropy-proposal/SKILL.md`
- Added `IDEA-009`, a Structural Entropy latent-transition abstraction.
- SE mapping:
  - `V`: rollout latent states from TD-MPC2 training batches.
  - `E`: consecutive predicted latent transitions in the model rollout.
  - `W`: soft transition flow between SimNorm latent symbols.
  - `2m`: total in-volume plus out-volume of the soft transition graph.
  - `T`: one-level soft encoding tree with root plus `simnorm_dim` symbolic modules.
  - `V_alpha`: soft module volume.
  - `g_alpha`: module cut volume, approximated as volume minus within-module transition flow.
- Code changes:
  - `tdmpc2/config.yaml`: added `se_coef`, default `0.0`.
  - `tdmpc2/tdmpc2.py`: added `_structural_entropy_loss` and logs `se_loss`.
  - `structural_entropy_proposal.md`: records the proposal, math, implementation plan, risks, and citations.
- Prepared run:
  - script: `scripts/autosota_iter8_acrobot_se_abstraction.sh`
  - task: `acrobot-swingup`
  - model size: `1`
  - steps: `400000`
  - seed: `1`
  - eval frequency: `100000`
  - eval episodes: `10`
  - `se_coef=0.01`
  - `compile=false`
- The run script waits for iteration 7 to complete before starting, to avoid GPU contention.
- Planned outputs:
  - official-format CSV: `results/tdmpc2-codex/acrobot-swingup_se_abstraction.csv`
  - comparison CSV: `results/tdmpc2-codex/acrobot-swingup_se_abstraction_compare.csv`

## 2026-04-28: Iteration 8 Vast.ai Remote Run Staged

- Used the requested `autosota-vastai-scheduler` skill to rent a disposable Vast.ai instance.
- Rented instance:
  - contract: `35778978`
  - GPU: `1x RTX 3090`
  - location: Argentina
  - estimated price: about `$0.152/hr`
  - label: `autosota-tdmpc2-se-acrobot`
- The instance on-start command waits for:
  - `/workspace/tdmpc2-se-remote/run_remote.sh`
  - `/workspace/.wandb_env`
- Uploaded non-secret payload:
  - `scripts/vastai_iter8_se_remote_run.sh` to `/workspace/tdmpc2-se-remote/run_remote.sh`
  - `logs/autosota/iter8_se_remote.patch` to `/workspace/tdmpc2-se-remote/se.patch`
- The instance is currently waiting for the W&B environment file.
- Secret handling note:
  - The W&B API key was not written to repository files, local logs, launch metadata, or the research log.
  - The remote job will start after `/workspace/.wandb_env` is created on the instance.

## 2026-04-28: Iteration 8 Vast.ai Direct Run Attempt

- The first rented instance, contract `35778978`, started from the uploaded patch payload but failed because the broad patch included documentation files whose context did not match the remote branch.
- The failed contract is no longer listed by Vast.ai, consistent with scheduler cleanup.
- Prepared a safer tarball payload:
  - `/tmp/tdmpc2-se-payload.tgz`
  - contains the prepared local TD-MPC2 workspace without `.git`, `logs`, `wandb`, or Python cache files.
  - remote runner: `scripts/vastai_iter8_se_remote_payload_run.sh`
- Rented replacement instance:
  - contract: `35781940`
  - GPU: `1x RTX 3090`
  - location: Belgium
  - estimated price: about `$0.171/hr`
  - label: `autosota-tdmpc2-se-acrobot-payload`
- Uploaded the tarball payload and W&B env file.
- Current remote status:
  - the on-start job has launched from the tarball.
  - dependency installation is in progress.
  - training has not yet reached first TD-MPC2 output at the time of this log update.
- Secret handling note:
  - The W&B key was used only to create `/workspace/.wandb_env` on the rented instance.
  - The key is not recorded in repository files or this research log.
