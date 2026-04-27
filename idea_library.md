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
