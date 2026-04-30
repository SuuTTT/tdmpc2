# Idea Library: TD-MPC2 Codex

### IDEA-001: Validate Short Baseline Loop
- **Granularity**: PARAM
- **Risk**: LOW
- **Admissibility**: CLEARED
- **Priority**: HIGH
- **Metric Target**: episode_reward
- **Lever**: `steps=1000`, `eval_episodes=1`
- **Evidence**: Needed to validate environment and scheduler lifecycle.
- **Hypothesis**: A short run will produce a checkpoint and measurable reward, enabling safe Scheduler iteration.
- **Protocol Audit**: Smoke tier only; not claimed as paper reproduction.
- **Implementation Sketch**: Run `train_command`, then `validation_command`.
- **Status**: SUCCESS
- **Result**: Baseline smoke run completed with 1000 training steps; one-episode validation reward was 6.3.

### IDEA-002: Increase Smoke Training Steps
- **Granularity**: PARAM
- **Risk**: LOW
- **Admissibility**: CLEARED
- **Priority**: MEDIUM
- **Metric Target**: episode_reward
- **Lever**: Increase `steps` after baseline loop works.
- **Evidence**: More environment interaction generally improves RL learning signal.
- **Hypothesis**: A modest step increase may improve reward while keeping iteration cost manageable.
- **Protocol Audit**: Valid within smoke tier if clearly recorded.
- **Implementation Sketch**: Test `steps=5000` or `steps=10000`.
- **Status**: FAILED
- **Result**: Tested a cheaper 2000-step variant first; one-episode validation reward was 6.3, matching baseline and not improving `_best`.

### IDEA-003: Tune Planning Horizon
- **Granularity**: PARAM
- **Risk**: MEDIUM
- **Admissibility**: CLEARED
- **Priority**: MEDIUM
- **Metric Target**: episode_reward
- **Lever**: `horizon` in `tdmpc2/config.yaml`.
- **Evidence**: Longer planning can help coordinated locomotion but increases compute.
- **Hypothesis**: Slightly longer horizon may improve early dog-run behavior.
- **Protocol Audit**: Evaluation remains unchanged; training/config change is admissible in optimization tier.
- **Implementation Sketch**: Compare default horizon against one nearby value after baseline.
- **Status**: PENDING
- **Result**: TBD

### IDEA-004: Human Prior - Latent Bottleneck Abstraction Probe
- **Origin**: Human prior
- **Granularity**: PARAM
- **Risk**: LOW
- **Admissibility**: CLEARED
- **Priority**: HIGH
- **Metric Target**: episode_reward
- **Lever**: `model_size=1` in TD-MPC2, which sets `latent_dim=128` instead of the `model_size=5` baseline latent dimension of 512.
- **Evidence**: User prior plus TD-MPC2's existing model-size abstraction knob. Related literature supports latent/discrete/hierarchical abstraction, but TD-MPC2 itself also benefits from scale, so this must be tested empirically.
- **Hypothesis**: A stronger latent bottleneck may improve early smoke-tier dog-run planning stability by filtering high-dimensional proprioceptive noise and reducing planner/model complexity.
- **Protocol Audit**: Evaluation task, reward, metric, and evaluation script are unchanged. This is a model-capacity/representation probe, not reward shaping or dataset leakage.
- **Implementation Sketch**: Train `task=dog-run model_size=1 steps=50000 seed=1 exp_name=iter2_abstraction_m1_steps50000 enable_wandb=false save_video=false compile=false`, then evaluate with the same one-episode smoke-tier command and absolute checkpoint path.
- **Status**: SUCCESS
- **Result**: Background run completed with one-episode reward 55.2 at 50000 steps. This is improved versus earlier short smoke baselines, but not yet a fair comparison against original `model_size=5` at the same 50000-step budget.

### IDEA-005: Human Prior - Discrete Latent Codebook Abstraction
- **Origin**: Human prior
- **Granularity**: ALGO
- **Risk**: HIGH
- **Admissibility**: REVIEW
- **Priority**: MEDIUM
- **Metric Target**: episode_reward
- **Lever**: Encoder/world-model latent representation in `tdmpc2/common/world_model.py` and `tdmpc2/common/layers.py`.
- **Evidence**: Discrete Codebook World Models report benefits of discrete stochastic latent states for continuous control and compare against TD-MPC2-style continuous latent models.
- **Hypothesis**: Adding a discrete/codebook bottleneck could improve control-relevant abstraction and planner robustness.
- **Protocol Audit**: Potentially admissible if evaluation semantics are unchanged, but high-risk because it changes the core world-model representation.
- **Implementation Sketch**: Prototype only after low-risk bottleneck and horizon probes; require AgentSupervisor review before execution.
- **Status**: PENDING
- **Result**: TBD

### IDEA-006: Human Prior - Temporal Abstraction Planner
- **Origin**: Human prior
- **Granularity**: ALGO
- **Risk**: HIGH
- **Admissibility**: REVIEW
- **Priority**: LOW
- **Metric Target**: episode_reward
- **Lever**: Planner in `tdmpc2/tdmpc2.py`, especially action sequence sampling and horizon semantics.
- **Evidence**: Latent temporal-abstraction planning work argues that primitive-timescale planning can be costly and brittle for long-horizon continuous control.
- **Hypothesis**: Planning over short action chunks or latent macro-actions may improve long-horizon dog-run coordination.
- **Protocol Audit**: Requires careful review because it changes planning semantics; reward and evaluation must remain frozen.
- **Implementation Sketch**: Defer until smoke baseline and lower-risk abstraction probes are complete.
- **Status**: PENDING
- **Result**: TBD

### IDEA-007: Planner-Cost Reduction for Faster Abstraction Runs
- **Origin**: Scheduler follow-up
- **Granularity**: PARAM
- **Risk**: MEDIUM
- **Admissibility**: CLEARED
- **Priority**: HIGH
- **Metric Target**: episode_reward and wall-clock runtime
- **Lever**: TD-MPC2 planner parameters: `num_samples`, `num_elites`, `num_pi_trajs`, and `iterations`.
- **Evidence**: `dog-run` action dimension triggers extra MPPI iterations in `tdmpc2/tdmpc2.py`; reducing planner samples and iterations should reduce action-selection cost during online training.
- **Hypothesis**: For early 50k-step search, `model_size=1 num_samples=256 num_elites=32 num_pi_trajs=12 iterations=4` may keep enough planning quality while materially reducing wall-clock runtime versus the default `model_size=1` abstraction run.
- **Protocol Audit**: Evaluation task, reward, metric, and evaluation script are unchanged. This is a planner-cost optimization, not a change to benchmark semantics.
- **Implementation Sketch**: Train `task=dog-run model_size=1 steps=50000 seed=1 exp_name=iter5_fastplanner_m1_steps50000 num_samples=256 num_elites=32 num_pi_trajs=12 iterations=4 enable_wandb=false save_video=false compile=false`, then evaluate with `eval_episodes=10`.
- **Status**: SUCCESS
- **Result**: Completed with reward 47.8 over 10 evaluation episodes and train time 1:18:17. This is slightly below the validated default `model_size=1` abstraction score of 50.5, but competitive and faster than the rough 1.5-hour default runtime estimate.

### IDEA-008: Information Bottleneck Latent Abstraction
- **Origin**: Human follow-up
- **Granularity**: ALGO
- **Risk**: MEDIUM
- **Admissibility**: CLEARED
- **Priority**: HIGH
- **Metric Target**: episode_reward
- **Lever**: Add `ib_coef` to the TD-MPC2 world-model update. The loss penalizes KL from each SimNorm latent group to a uniform categorical prior.
- **Evidence**: Information bottleneck methods encourage representations that preserve task-relevant predictive information while discarding nuisance detail. TD-MPC2 already uses grouped SimNorm latent probabilities, which gives a low-friction place to apply a KL-to-prior abstraction penalty.
- **Hypothesis**: A small information-bottleneck coefficient may improve cross-task abstraction by regularizing the latent model beyond the simple `model_size=1` bottleneck.
- **Protocol Audit**: Evaluation task, reward, and environment remain unchanged. This modifies the learned representation objective, so results should be compared against a same-task, same-step `model_size=1` run.
- **Implementation Sketch**: Train `task=acrobot-swingup model_size=1 steps=400000 seed=1 eval_freq=100000 eval_episodes=10 ib_coef=0.01 compile=false exp_name=iter7_acrobot_ib_m1_steps400000`.
- **Status**: READY
- **Result**: Code and run script prepared. The run script waits for the active iteration 6 acrobot compile job before starting to avoid GPU contention.

### IDEA-009: Structural Entropy Latent Transition Abstraction
- **Origin**: Human follow-up via `structural-entropy-proposal` skill
- **Granularity**: ALGO
- **Risk**: MEDIUM
- **Admissibility**: CLEARED
- **Priority**: HIGH
- **Metric Target**: episode_reward
- **Lever**: Add `se_coef` to the TD-MPC2 world-model update. The loss builds a soft transition-flow graph over rollout latents and minimizes a one-level structural entropy proxy.
- **Evidence**: Structural Entropy models uncertainty in graph/network structure through encoding trees. TD-MPC2 already learns latent transition dynamics, giving a natural graph: latent states as vertices and predicted transitions as edges.
- **Hypothesis**: Encouraging low structural entropy in the latent transition flow can create more coherent macro-state modules, improving planning and early sample efficiency beyond capacity-only abstraction.
- **Protocol Audit**: Evaluation task, reward, and environment remain unchanged. This changes representation training, so it must be compared against same-task, same-step `model_size=1` probes.
- **Implementation Sketch**: Train `task=acrobot-swingup model_size=1 steps=400000 seed=1 eval_freq=100000 eval_episodes=10 se_coef=0.01 compile=false exp_name=iter8_acrobot_se_m1_steps400000`.
- **Status**: READY
- **Result**: Code, proposal, and run script prepared. The run script waits for iteration 7 to complete before starting to avoid GPU contention.
