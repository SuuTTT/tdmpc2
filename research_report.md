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

## Human Prior: Abstraction for TD-MPC2
- **User idea**: The user hypothesizes that abstraction can enhance TD-MPC2.
- **Mechanism**: TD-MPC2 already plans in a learned latent world model, so useful abstraction could mean better latent compression, temporal abstraction, discrete/codebook latent structure, or hierarchical planning over coarser dynamics. For `dog-run`, the plausible benefit is reducing noisy proprioceptive detail and improving planning stability for high-dimensional locomotion.
- **TD-MPC2 grounding**: TD-MPC2 performs local trajectory optimization in the latent space of an implicit decoder-free world model and reports strong transfer across many continuous-control tasks. Its built-in `MODEL_SIZE` table provides a low-risk latent bottleneck probe: `model_size=1` uses `latent_dim=128`, while `model_size=5` uses `latent_dim=512`.
- **External evidence**: Recent latent world-model work supports abstraction as a serious direction. Hierarchical latent planning argues for multi-scale latent world models to reduce long-horizon planning error and planning-time compute. Discrete Codebook World Models compare against TD-MPC2 and argue that discrete latent state abstractions can be effective for continuous control. In-Context Planning with Latent Temporal Abstractions frames primitive-timescale planning as expensive and uses learned discrete macro-action tokens for planning.
- **Counter-evidence or risks**: TD-MPC2's paper also reports that capability increases with model/data size, so a smaller latent model may underfit. More invasive abstraction ideas, such as reward shaping, environment changes, or replacing the planner with a different algorithm, require AgentSupervisor review.
- **Executable probes**: IDEA-004 tests latent bottleneck abstraction through `model_size=1`; IDEA-005 and IDEA-006 remain higher-risk follow-ups.

### Sources
- TD-MPC2 paper: https://arxiv.org/abs/2310.16828
- TD-MPC2 OpenReview summary: https://openreview.net/forum?id=FzpfPa6unv
- Hierarchical Planning with Latent World Models: https://arxiv.org/abs/2604.03208
- Discrete Codebook World Models for Continuous Control: https://arxiv.org/abs/2503.00653
- In-Context Planning with Latent Temporal Abstractions: https://arxiv.org/abs/2602.18694
