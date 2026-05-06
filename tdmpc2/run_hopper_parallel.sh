#!/bin/bash
export MUJOCO_GL=egl
export WANDB_SILENT=true

# 1. Setup Environment
chmod +x setup-env-tdmpc2.sh
./setup-env-tdmpc2.sh

# 2. Launch 1D SE Hopper Run (Most promising: se_coef=0.1, ib_coef=0.1)
python3 train.py \
    task=hopper-hop \
    model_size=1 \
    steps=4000000 \
    eval_freq=50000 \
    se_coef=0.1 \
    ib_coef=0.1 \
    wandb_entity=sudingli21 \
    wandb_project=tdmpc2-codex \
    exp_name=hopper_1dse_4m \
    compile=false \
    save_video=false &

# 3. Launch 2D SE Hopper Run (Most promising: se_coef=0.1, ib_coef=0.1, super_modules=2)
python3 train_2dse.py \
    task=hopper-hop \
    model_size=1 \
    steps=4000000 \
    eval_freq=50000 \
    se_coef=0.1 \
    ib_coef=0.1 \
    +se_2d=true \
    +num_super_modules=2 \
    wandb_entity=sudingli21 \
    wandb_project=tdmpc2-codex \
    exp_name=hopper_2dse_4m \
    compile=false \
    save_video=false &

# Wait for both to finish
wait
echo "All Hopper runs completed."
