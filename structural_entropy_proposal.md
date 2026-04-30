# Structural Entropy Proposal: TD-MPC2 Latent Transition Abstraction

## Baseline

TD-MPC2 learns a latent world model and plans through predicted latent transitions. The current abstraction evidence comes from `model_size=1`, which reduces latent capacity, and from an information-bottleneck probe. Neither explicitly models the structure of state-transition flow.

## SE Formulation

- `V`: rollout latent states `z_t` sampled from TD-MPC2 training batches.
- `E`: predicted one-step latent transitions `(z_t, z_{t+1})` over the model rollout horizon.
- `W`: soft transition flow induced by SimNorm symbol assignments.
- `2m`: total directed transition volume, implemented as total in-volume plus out-volume over soft modules.
- `T`: a one-level soft encoding tree. The root contains all rollout latent states; children are `simnorm_dim` symbolic modules. Each latent state belongs softly to modules by averaging its SimNorm group probabilities.
- `V_alpha`: soft in-volume plus out-volume of module `alpha`.
- `g_alpha`: soft cut volume of module `alpha`, computed as module volume minus within-module transition flow.

The implemented objective is a differentiable one-level structural entropy proxy:

```text
H_T = - sum_alpha (g_alpha / 2m) log(V_alpha / 2m)
```

The training loss becomes:

```text
L = L_TD-MPC2 + se_coef * H_T
```

## Implementation

- `tdmpc2/config.yaml`: adds `se_coef`, default `0.0`.
- `tdmpc2/tdmpc2.py`: adds `_structural_entropy_loss(zs)` and logs `se_loss`.
- Default behavior is unchanged when `se_coef=0.0`.
- The first probe uses `se_coef=0.01` on `acrobot-swingup`.

## Experiment

Run a short same-task probe after the active GPU jobs:

- task: `acrobot-swingup`
- model size: `1`
- steps: `400000`
- seed: `1`
- eval frequency: `100000`
- eval episodes: `10`
- `compile=false`
- `se_coef=0.01`

Compare against:

- official TD-MPC2 CSV: `results/tdmpc2/acrobot-swingup.csv`
- current abstraction compile probe at matching early milestones
- information-bottleneck probe if it completes first

## Risks

- The one-level soft tree can over-cluster latent symbols if `se_coef` is too large.
- Batch-local transition structure may be noisy early in training.
- This proxy uses SimNorm symbol modules, not a full SE-optimized discrete tree.

Mitigations:

- Keep `se_coef` small.
- Use official-format CSV comparison at matching steps.
- Treat this as an inexpensive falsification probe before implementing offline replay-graph hierarchy discovery.

## References

- Li, A., and Pan, Y. Structural information and dynamical complexity of networks. IEEE Transactions on Information Theory, 2016.
- Rosvall, M., and Bergstrom, C. T. Maps of random walks on complex networks reveal community structure. PNAS, 2008.
- Su, D., et al. A Survey of Structural Entropy: Theory, Methods, and Applications. IJCAI, 2025.
