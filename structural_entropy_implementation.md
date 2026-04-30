# Structural Entropy Implementation for TD-MPC2 Representation Learning

This document describes the structural entropy (SE) integration that was added to TD-MPC2 as a representation regularizer. It is an implementation note, not just a proposal: the code path is active behind `se_coef`, and the default value keeps baseline TD-MPC2 behavior unchanged.

## Goal

The abstraction idea is to bias the TD-MPC2 latent representation toward reusable transition structure. TD-MPC2 already learns a latent world model and optimizes consistency, reward, value, and policy losses. The SE addition adds one more differentiable term over the latent rollout graph:

```text
L_total = L_TD-MPC2 + ib_coef * L_IB + se_coef * L_SE
```

For the SE run, `ib_coef` remains `0.0` and `se_coef` is set to `0.01`.

## Files Changed

- `tdmpc2/config.yaml`
  - Adds `ib_coef: 0.0`
  - Adds `se_coef: 0.0`
- `tdmpc2/tdmpc2.py`
  - Adds `_latent_ib_loss(z)`
  - Adds `_structural_entropy_loss(zs)`
  - Stores latent rollout states in `zs`
  - Adds `self.cfg.se_coef * se_loss` to `total_loss`
  - Logs `se_loss`
- `scripts/autosota_iter8_acrobot_se_abstraction.sh`
  - Local SE run wrapper
- `scripts/vastai_iter8_se_remote_payload_run.sh`
  - Vast.ai SE run wrapper
- `structural_entropy_proposal.md`
  - Short research proposal and experiment framing

## Representation View

TD-MPC2 uses SimNorm latent vectors. A latent vector can be viewed as several simplex groups:

```text
z shape:                  [..., latent_dim]
reshaped z:               [..., num_groups, simnorm_dim]
soft symbol assignment:   [..., simnorm_dim]
```

The SE implementation uses the SimNorm structure as a soft symbolic abstraction. Instead of creating a hard cluster ID, it averages the probabilities across SimNorm groups:

```python
assignments = zs.view(*zs.shape[:-1], -1, self.cfg.simnorm_dim).mean(-2)
```

Each rollout latent state therefore has a soft membership vector over `simnorm_dim` modules. With the default small model used in the current acrobot probe, this means the latent state is softly assigned over 8 symbolic modules.

## Latent Transition Graph

During `_update`, TD-MPC2 already rolls the model forward across the planning horizon. The implementation now stores those rollout latents:

```python
zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
z = self.model.encode(obs[0], task)
zs[0] = z

for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
    z = self.model.next(z, _action, task)
    zs[t+1] = z
```

This creates a small batch-local directed graph:

- vertices: latent states in `zs`
- directed edges: consecutive rollout transitions `z_t -> z_{t+1}`
- modules: soft SimNorm symbol assignments
- edge weights: expected transition flow between source and target modules

The SE term is computed on model-predicted rollout latents, not on a separate offline replay graph. This keeps the first implementation cheap and differentiable.

## Structural Entropy Proxy

The implementation is a one-level soft structural entropy proxy. It does not construct a full multilevel SE coding tree. The root contains all rollout latent states; children are the SimNorm symbolic modules.

The code path is:

```python
def _structural_entropy_loss(self, zs):
    assignments = zs.view(*zs.shape[:-1], -1, self.cfg.simnorm_dim).mean(-2)
    source = assignments[:-1].reshape(-1, self.cfg.simnorm_dim)
    target = assignments[1:].reshape(-1, self.cfg.simnorm_dim)
    flow = source.transpose(0, 1).matmul(target) / source.shape[0]
    volume = flow.sum(0) + flow.sum(1)
    internal = flow.diag()
    cut = (volume - 2 * internal).clamp_min(0)
    total_volume = volume.sum().clamp_min(1e-8)
    ratio = (volume / total_volume).clamp_min(1e-8)
    return -(cut / total_volume * ratio.log()).sum()
```

The formula corresponds to:

```text
H_SE = - sum_alpha (g_alpha / vol_total) * log(vol_alpha / vol_total)
```

Where:

- `alpha` is a SimNorm symbolic module.
- `flow[i, j]` is the soft transition mass from module `i` to module `j`.
- `volume[alpha]` is incoming plus outgoing flow for module `alpha`.
- `internal[alpha]` is within-module flow, the diagonal of the transition matrix.
- `cut[alpha]` is module boundary flow: `volume - 2 * internal`.
- `total_volume` normalizes all module volumes.

Minimizing this term favors latent assignments where transition flow is compressible: high-volume modules with lower boundary flow become cheaper under the code-length objective.

## Why SimNorm Modules

A full SE graph partitioner would require building and optimizing a graph over many replay states, then coupling that graph back into online representation learning. For the first test, SimNorm gives an existing differentiable symbolic basis:

- It is already part of the TD-MPC2 latent representation.
- It provides normalized soft memberships.
- It avoids hard clustering during training.
- It keeps the loss differentiable end to end.
- It adds no new model parameters.

This makes SE a lightweight representation regularizer rather than a separate planner or offline abstraction pipeline.

## Loss Integration

The SE loss is computed once per TD-MPC2 model update after latent rollout construction:

```python
se_loss = self._structural_entropy_loss(zs)
```

It is added to the existing world-model loss:

```python
total_loss = (
    self.cfg.consistency_coef * consistency_loss +
    self.cfg.reward_coef * reward_loss +
    self.cfg.termination_coef * termination_loss +
    self.cfg.value_coef * value_loss +
    self.cfg.ib_coef * ib_loss +
    self.cfg.se_coef * se_loss
)
```

The policy update receives `zs.detach()` as before:

```python
pi_info = self.update_pi(zs.detach(), task)
```

So the SE term affects representation and world-model learning through the model optimizer. It does not directly backpropagate through the policy update.

## Default Behavior

Baseline behavior is preserved because:

```yaml
ib_coef: 0.0
se_coef: 0.0
```

When `se_coef=0.0`, `se_loss` is still computed and logged, but it contributes zero to `total_loss`. This keeps the config compatible with baseline runs and makes it easy to inspect the diagnostic value of `se_loss` before enabling the regularizer.

## Current SE Experiment

The active SE probe uses:

```text
task: acrobot-swingup
model_size: 1
seed: 1
steps: 400000
eval_freq: 100000
eval_episodes: 10
compile: false
se_coef: 0.01
wandb_project: tdmpc2-codex
```

The live Vast.ai run writes its raw eval CSV here on the remote instance:

```text
/workspace/tdmpc2-codex/logs/acrobot-swingup/1/vastai_iter8_acrobot_se_m1_steps400000/eval.csv
```

A partial copy was exported locally here:

```text
results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai_partial.csv
results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai_partial_compare.csv
```

The earlier compile run is separate:

```text
results/tdmpc2-codex/acrobot-swingup_abstraction_compile_compare.csv
```

That file is from the 4,000,000-step compile-abstraction run, not the current 400,000-step SE run.

## Current Partial Result

At the last local pull, the current SE run had evaluated through 300k steps:

```csv
step,reward,seed
0,9.1,1
100000,374.9,1
200000,390.6,1
300000,564.7,1
```

The matching official-format comparison is:

```csv
step,official_tdmpc2_mean_reward,abstraction_compile_reward,delta,seed
0,5.2,9.1,3.9,1
100000,179.1,374.9,195.8,1
200000,296.3,390.6,94.3,1
300000,346.9,564.7,217.8,1
```

The exporter column name still says `abstraction_compile_reward`; that is a naming artifact in the generic exporter and should be renamed before using these CSVs in a paper-style table.

## Interpretation

The implemented SE loss is best understood as a transition-flow regularizer over latent symbols. It asks the representation to make predicted rollout transitions easier to describe with a shallow symbolic code.

Expected benefit:

- similar transition modes reuse the same soft symbols;
- high-frequency transition flow is encoded compactly;
- model-size-1 representations may become less noisy and more reusable.

Expected risk:

- too large `se_coef` can collapse distinct dynamics into the same symbol;
- batch-local transition flow can be noisy early in training;
- one-level SE may miss useful hierarchical structure.

The first run uses `se_coef=0.01` to keep the regularizer small relative to TD-MPC2's main objectives.

## Limitations

This is not a full structural entropy tree learner. It does not:

- build a replay-wide transition graph;
- optimize a multilevel coding tree;
- learn hard discrete state partitions;
- modify MPC planning directly;
- add a separate abstraction model.

It is deliberately a minimal differentiable proxy so the idea can be falsified quickly against official TD-MPC2 CSV curves.

## Follow-Up Improvements

Useful next changes:

- Rename exported comparison columns so SE runs do not report `abstraction_compile_reward`.
- Log `se_loss` alongside reward in W&B charts for direct inspection.
- Sweep `se_coef` over `{0.001, 0.003, 0.01, 0.03}`.
- Compare against the IB-only run at the same steps and model size.
- Add a replay-buffer graph analysis after training to measure whether learned latents actually reduce transition coding entropy.
- Try a two-level coding tree: coarse modules from transition communities, fine modules from SimNorm symbols.

