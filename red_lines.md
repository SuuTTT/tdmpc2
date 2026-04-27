# Red Lines: TD-MPC2

- Do not modify evaluation metric semantics in `tdmpc2/evaluate.py`.
- Do not change the DMControl task definition or reward function to inflate scores.
- Do not use ground-truth future trajectories or privileged evaluation information.
- Do not compare smoke-run scores to full paper training results as if equivalent.
- Do not accept a new best state unless it is evaluated with the same command as the measured baseline for that run tier.
- Keep dataset/task split semantics unchanged; `dog-run` remains the target task for this initial loop.
