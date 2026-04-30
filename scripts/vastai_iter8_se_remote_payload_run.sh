#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT=true

REPO=/workspace/tdmpc2-codex
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -x /opt/conda/bin/python ]]; then
    PYTHON_BIN=/opt/conda/bin/python
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys, torch
print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

apt-get update
apt-get install -y --no-install-recommends \
  build-essential git rsync curl wget ffmpeg swig \
  libjpeg-dev libpng-dev libssl-dev libcurl4-openssl-dev \
  zlib1g-dev libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev \
  libvulkan1 libgl1-mesa-glx libosmesa6 libosmesa6-dev \
  libglew-dev mesa-utils
rm -rf /var/lib/apt/lists/*

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install \
  dm-control==1.0.16 \
  glfw==2.7.0 \
  gymnasium==0.29.1 \
  ffmpeg==1.4 \
  imageio==2.34.1 \
  imageio-ffmpeg==0.4.9 \
  h5py==3.11.0 \
  hydra-core==1.3.2 \
  hydra-submitit-launcher==1.2.0 \
  submitit==1.5.1 \
  omegaconf==2.3.0 \
  moviepy==1.0.3 \
  mujoco==3.1.2 \
  numpy==1.24.4 \
  tensordict-nightly==2025.1.1 \
  torchrl-nightly==2025.1.1 \
  kornia==0.7.2 \
  termcolor==2.4.0 \
  tqdm==4.66.4 \
  pandas==2.0.3 \
  wandb==0.17.4

cd "$REPO"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set" >&2
  exit 2
fi
"$PYTHON_BIN" -m wandb login --relogin "$WANDB_API_KEY"

EXP_NAME=vastai_iter8_acrobot_se_m1_steps400000
TASK=acrobot-swingup
SEED=1
STEPS=400000
EVAL_EPISODES=10
EVAL_FREQ=100000

"$PYTHON_BIN" tdmpc2/train.py \
  task="$TASK" \
  model_size=1 \
  steps="$STEPS" \
  seed="$SEED" \
  eval_freq="$EVAL_FREQ" \
  eval_episodes="$EVAL_EPISODES" \
  exp_name="$EXP_NAME" \
  se_coef=0.01 \
  enable_wandb=true \
  wandb_project=tdmpc2-codex \
  wandb_entity=null \
  wandb_silent=true \
  save_video=false \
  compile=false

"$PYTHON_BIN" scripts/export_acrobot_compile_result.py \
  --official results/tdmpc2/acrobot-swingup.csv \
  --eval-csv "logs/${TASK}/${SEED}/${EXP_NAME}/eval.csv" \
  --result-csv results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai.csv \
  --compare-csv results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai_compare.csv \
  --scores-jsonl scores.jsonl \
  --state-json .autosota_scheduler_test/iter8_acrobot_se_abstraction_vastai_state.json \
  --run-log "/workspace/vastai-job/job.log" \
  --checkpoint "logs/${TASK}/${SEED}/${EXP_NAME}/models/final.pt" \
  --task "$TASK" \
  --iteration 8 \
  --mode structural_entropy_abstraction_vastai \
  --idea-id IDEA-009 \
  --notes "Vast.ai acrobot-swingup structural-entropy abstraction probe over soft latent transition flow." \
  --seed "$SEED" \
  --model-size 1 \
  --steps "$STEPS" \
  --eval-episodes "$EVAL_EPISODES" \
  --pre-commit "$(git rev-parse HEAD 2>/dev/null || echo payload)"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json
import wandb

run = wandb.init(project="tdmpc2-codex", entity=None, name="vastai_iter8_acrobot_se_export", job_type="export")
paths = [
    Path("results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai.csv"),
    Path("results/tdmpc2-codex/acrobot-swingup_se_abstraction_vastai_compare.csv"),
    Path(".autosota_scheduler_test/iter8_acrobot_se_abstraction_vastai_state.json"),
    Path("scores.jsonl"),
]
for path in paths:
    if path.exists():
        wandb.save(str(path))
state_path = Path(".autosota_scheduler_test/iter8_acrobot_se_abstraction_vastai_state.json")
if state_path.exists():
    state = json.loads(state_path.read_text())
    for key in ("reward", "official_mean_reward_at_final_step", "train_time"):
        if key in state:
            run.summary[key] = state[key]
run.finish()
PY

echo "TDMPC2_SE_REMOTE_DONE $(date -Is)"
