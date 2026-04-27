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
