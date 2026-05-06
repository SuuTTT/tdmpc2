#!/bin/bash
export MUJOCO_GL=egl
export WANDB_SILENT=true

# 1. Setup Environment
chmod +x setup-env-tdmpc2.sh
./setup-env-tdmpc2.sh

# 2. Pre-flight check
PYTHON_EXEC=$(which python3)
echo "Running pre-flight dependency check using $PYTHON_EXEC..."
$PYTHON_EXEC -c "import matplotlib.pyplot as plt; print('Matplotlib is ready.')"

# 3. Launch Hopper Parallel (Long runs)
echo "Launching Hopper runs in background..."
$PYTHON_EXEC train.py \
    task=hopper-hop \
    model_size=1 \
    steps=4000000 \
    eval_freq=50000 \
    se_coef=0.1 \
    ib_coef=0.1 \
    wandb_entity=sudingli21 \
    wandb_project=tdmpc2-codex \
    exp_name=hopper_1dse_4m_final \
    compile=false \
    save_video=false &

$PYTHON_EXEC train_2dse.py \
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
    exp_name=hopper_2dse_4m_final \
    compile=false \
    save_video=false &

# 4. Run Acrobot Sweep (Fast runs, sequential)
echo "Starting Acrobot sweep in parallel foreground..."
ib_values=("0.01" "0.05" "0.1")
for ib in "${ib_values[@]}"; do
    echo "------------------------------------------------"
    echo "Starting Acrobot run with ib_coef=$ib"
    echo "------------------------------------------------"
    $PYTHON_EXEC train.py \
        task=acrobot-swingup \
        model_size=1 \
        steps=100000 \
        eval_freq=10000 \
        se_coef=0.01 \
        ib_coef=$ib \
        wandb_entity=sudingli21 \
        wandb_project=tdmpc2-codex \
        exp_name=acrobot_se0.01_ib${ib}_sweep_final \
        compile=false \
        save_video=false
done

# Wait for background jobs
wait
echo "All experiments completed."
