#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT=true

WORK=/workspace/tdmpc2-se-remote
REPO=/workspace/tdmpc2-codex
PAYLOAD="$WORK/payload.tgz"
ENV_FILE=/workspace/.wandb_env
PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"

mkdir -p "$WORK"
cd "$WORK"

echo "waiting_for_payload $(date -Is)"
for _ in $(seq 1 720); do
  [[ -s "$PAYLOAD" && -s "$ENV_FILE" ]] && break
  sleep 10
done
if [[ ! -s "$PAYLOAD" ]]; then
  echo "payload_missing" >&2
  exit 3
fi
if [[ ! -s "$ENV_FILE" ]]; then
  echo "wandb_env_missing" >&2
  exit 4
fi

set -a
source "$ENV_FILE"
set +a

if [[ ! -x "$PYTHON_BIN" ]]; then
  if [[ -x /opt/conda/bin/python ]]; then
    PYTHON_BIN=/opt/conda/bin/python
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
import torch
print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("arch_list", torch.cuda.get_arch_list() if torch.cuda.is_available() else None)
PY

set +e
"$PYTHON_BIN" - <<'PY'
import torch
needs = torch.cuda.is_available() and any("RTX 50" in torch.cuda.get_device_name(i) or "RTX 5060" in torch.cuda.get_device_name(i) or "RTX 5070" in torch.cuda.get_device_name(i) or "RTX 5080" in torch.cuda.get_device_name(i) or "RTX 5090" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
arch = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
if needs and not any(x in arch for x in ("sm_120", "compute_120")):
    raise SystemExit(42)
PY
probe_status=$?
set -e
if [[ "$probe_status" -eq 42 ]]; then
  echo "installing_pytorch_nightly_for_rtx50 $(date -Is)"
  "$PYTHON_BIN" -m pip install --upgrade --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu132
elif [[ "$probe_status" -ne 0 ]]; then
  exit "$probe_status"
fi

"$PYTHON_BIN" - <<'PY'
import sys
import torch
print("post_torch", torch.__version__, torch.version.cuda)
print("post_cuda_available", torch.cuda.is_available())
print("post_cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("post_arch_list", torch.cuda.get_arch_list() if torch.cuda.is_available() else None)
PY

apt-get update
apt-get install -y --no-install-recommends \
  build-essential git rsync curl wget ffmpeg swig \
  libjpeg-dev libpng-dev libssl-dev libcurl4-openssl-dev \
  zlib1g-dev libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev \
  libvulkan1 libosmesa6 libosmesa6-dev \
  libglew-dev mesa-utils
rm -rf /var/lib/apt/lists/*

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install \
  dm-control \
  glfw \
  gymnasium==0.29.1 \
  ffmpeg \
  imageio \
  imageio-ffmpeg \
  h5py \
  hydra-core==1.3.2 \
  hydra-submitit-launcher \
  submitit \
  omegaconf \
  moviepy==1.0.3 \
  mujoco \
  "numpy>=1.26" \
  tensordict \
  torchrl \
  kornia \
  termcolor \
  tqdm \
  pandas \
  matplotlib \
  seaborn \
  wandb

rm -rf "$REPO"
mkdir -p "$REPO"
tar -xzf "$PAYLOAD" -C "$REPO"
cd "$REPO"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set" >&2
  exit 2
fi
"$PYTHON_BIN" -m wandb login --relogin "$WANDB_API_KEY"

EXP_NAME=vastai_rerun_acrobot_se_m1_steps400000
TASK=acrobot-swingup
SEED=1
STEPS=400000
EVAL_EPISODES=10
EVAL_FREQ=100000
SE_COEF=0.01
CHECKPOINT="logs/${TASK}/${SEED}/${EXP_NAME}/models/final.pt"
DIAG_DIR="results/tdmpc2-codex/${EXP_NAME}_diagnostics"

echo "starting_train $(date -Is)"
"$PYTHON_BIN" tdmpc2/train.py \
  task="$TASK" \
  model_size=1 \
  steps="$STEPS" \
  seed="$SEED" \
  eval_freq="$EVAL_FREQ" \
  eval_episodes="$EVAL_EPISODES" \
  exp_name="$EXP_NAME" \
  se_coef="$SE_COEF" \
  enable_wandb=true \
  wandb_project=tdmpc2-codex \
  wandb_entity=null \
  wandb_silent=true \
  save_video=false \
  save_agent=true \
  compile=false

test -s "$CHECKPOINT"

"$PYTHON_BIN" scripts/export_acrobot_compile_result.py \
  --official results/tdmpc2/acrobot-swingup.csv \
  --eval-csv "logs/${TASK}/${SEED}/${EXP_NAME}/eval.csv" \
  --result-csv results/tdmpc2-codex/acrobot-swingup_se_abstraction_rerun.csv \
  --compare-csv results/tdmpc2-codex/acrobot-swingup_se_abstraction_rerun_compare.csv \
  --scores-jsonl scores.jsonl \
  --state-json .autosota_scheduler_test/iter9_acrobot_se_abstraction_rerun_state.json \
  --run-log "/workspace/vastai-job/job.log" \
  --checkpoint "$CHECKPOINT" \
  --task "$TASK" \
  --iteration 9 \
  --mode structural_entropy_abstraction_rerun \
  --idea-id IDEA-009 \
  --notes "Rerun of Acrobot structural-entropy latent-transition abstraction with checkpoint and post-hoc structural diagnosis." \
  --seed "$SEED" \
  --model-size 1 \
  --steps "$STEPS" \
  --eval-episodes "$EVAL_EPISODES" \
  --pre-commit "$(git rev-parse HEAD 2>/dev/null || echo payload)"

echo "starting_structural_diagnosis $(date -Is)"
"$PYTHON_BIN" scripts/diagnose_structural_entropy.py \
  --checkpoint "$CHECKPOINT" \
  --task "$TASK" \
  --model-size 1 \
  --episodes 10 \
  --seed "$SEED" \
  --output-dir "$DIAG_DIR"

tar -czf "results/tdmpc2-codex/${EXP_NAME}_artifacts.tgz" \
  "$CHECKPOINT" \
  "logs/${TASK}/${SEED}/${EXP_NAME}/eval.csv" \
  "results/tdmpc2-codex/acrobot-swingup_se_abstraction_rerun.csv" \
  "results/tdmpc2-codex/acrobot-swingup_se_abstraction_rerun_compare.csv" \
  ".autosota_scheduler_test/iter9_acrobot_se_abstraction_rerun_state.json" \
  "$DIAG_DIR"

SUMMARY_JSON="results/tdmpc2-codex/${EXP_NAME}_wandb_summary.json"
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json

summary = {}
state_path = Path(".autosota_scheduler_test/iter9_acrobot_se_abstraction_rerun_state.json")
if state_path.exists():
    state = json.loads(state_path.read_text())
    for key in ("reward", "official_mean_reward_at_final_step", "train_time"):
        if key in state:
            summary[key] = state[key]
diag_dir = Path("results/tdmpc2-codex/vastai_rerun_acrobot_se_m1_steps400000_diagnostics")
diag_path = diag_dir / "structural_diagnostics.json"
if diag_path.exists():
    diag = json.loads(diag_path.read_text())
    summary["diagnostic_mean_episode_reward"] = diag.get("mean_episode_reward")
    for flow_key in ("predicted_transition_flow", "observed_encoded_transition_flow"):
        for metric, value in diag.get(flow_key, {}).items():
            summary[f"{flow_key}/{metric}"] = value
Path("results/tdmpc2-codex/vastai_rerun_acrobot_se_m1_steps400000_wandb_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

"$PYTHON_BIN" scripts/wandb_attach_artifacts.py \
  --project tdmpc2-codex \
  --run-dir "logs/${TASK}/${SEED}/${EXP_NAME}" \
  --name "$EXP_NAME" \
  --summary-json "$SUMMARY_JSON" \
  --artifact "results/tdmpc2-codex/acrobot-swingup_se_abstraction_rerun.csv" \
  --artifact "results/tdmpc2-codex/acrobot-swingup_se_abstraction_rerun_compare.csv" \
  --artifact ".autosota_scheduler_test/iter9_acrobot_se_abstraction_rerun_state.json" \
  --artifact "scores.jsonl" \
  --artifact "results/tdmpc2-codex/${EXP_NAME}_artifacts.tgz" \
  --artifact "$SUMMARY_JSON" \
  --artifact "$DIAG_DIR/structural_diagnostics.json" \
  --artifact "$DIAG_DIR/structural_diagnostics.csv" \
  --artifact "$DIAG_DIR/transition_graph_edges.csv" \
  --artifact "$DIAG_DIR/transition_graph_nodes.csv"

echo "TDMPC2_SE_TRAIN_DIAGNOSE_DONE $(date -Is)"
