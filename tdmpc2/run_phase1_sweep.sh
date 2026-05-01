#!/bin/bash
export MUJOCO_GL=egl
export WANDB_SILENT=true

# Wait for wandb login (if any) or run directly
ib_values=("0.01" "0.05" "0.1")

for ib in "${ib_values[@]}"; do
    echo "------------------------------------------------"
    echo "Starting run with ib_coef=$ib"
    echo "------------------------------------------------"
    python3 train.py \
        task=acrobot-swingup \
        model_size=1 \
        steps=100000 \
        eval_freq=10000 \
        se_coef=0.01 \
        ib_coef=$ib \
        wandb_entity=sudingli21 \
        wandb_project=tdmpc2-codex \
        exp_name=acrobot_se0.01_ib${ib}_sweep \
        compile=false \
        save_video=false
done
