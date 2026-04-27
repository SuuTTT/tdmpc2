#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/tdmpc2-codex
ENV_NAME=tdmpc2
ITER=3
IDEA_ID=COMPARATOR-M5-50000
EXP_NAME=iter3_original_m5_steps50000
TASK=dog-run
MODEL_SIZE=5
STEPS=50000
SEED=1
ABSTRACTION_REWARD=55.2

cd "$REPO"
mkdir -p logs/autosota
echo "$$" > "logs/autosota/${EXP_NAME}.pid"

export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PRE_COMMIT="$(git rev-parse HEAD)"
RUN_LOG="logs/autosota/${EXP_NAME}.log"
EVAL_LOG="logs/autosota/${EXP_NAME}_eval.log"
CHECKPOINT="$REPO/logs/${TASK}/${SEED}/${EXP_NAME}/models/final.pt"

{
  echo "timestamp=${START_TS}"
  echo "pre_commit=${PRE_COMMIT}"
  echo "idea_id=${IDEA_ID}"
  echo "task=${TASK}"
  echo "model_size=${MODEL_SIZE}"
  echo "steps=${STEPS}"
  echo "checkpoint=${CHECKPOINT}"
} > "logs/autosota/${EXP_NAME}_meta.txt"

echo "starting_train $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"
set +e
mamba run -n "$ENV_NAME" python tdmpc2/train.py \
  task="$TASK" \
  model_size="$MODEL_SIZE" \
  steps="$STEPS" \
  seed="$SEED" \
  exp_name="$EXP_NAME" \
  enable_wandb=false \
  save_video=false \
  compile=false 2>&1 | tee -a "$RUN_LOG"
train_status=${PIPESTATUS[0]}
set -e
echo "train_exit=${train_status} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

echo "starting_eval $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$EVAL_LOG"
set +e
mamba run -n "$ENV_NAME" python tdmpc2/evaluate.py \
  task="$TASK" \
  model_size="$MODEL_SIZE" \
  checkpoint="$CHECKPOINT" \
  eval_episodes=1 \
  save_video=false \
  compile=false 2>&1 | tee -a "$EVAL_LOG"
eval_status=${PIPESTATUS[0]}
set -e
echo "eval_exit=${eval_status} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$EVAL_LOG"
if [ "$eval_status" -ne 0 ]; then
  exit "$eval_status"
fi

python - <<'PY'
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo = Path("/workspace/tdmpc2-codex")
eval_log = repo / "logs/autosota/iter3_original_m5_steps50000_eval.log"
text = eval_log.read_text(encoding="utf-8", errors="replace")
matches = re.findall(r"R:\s*([-+]?\d+(?:\.\d+)?)\s+S:\s*([-+]?\d+(?:\.\d+)?)", text)
if not matches:
    raise SystemExit("Could not parse evaluation reward from eval log")
reward, success = map(float, matches[-1])
abstraction_reward = 55.2
decision = "ORIGINAL_BEATS_ABSTRACTION" if reward > abstraction_reward else "ABSTRACTION_REMAINS_AHEAD"
now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
pre_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

record = {
    "iteration": 3,
    "mode": "matched_original_comparator",
    "idea_id": "COMPARATOR-M5-50000",
    "granularity": "PARAM",
    "task": "dog-run",
    "metric": "episode_reward",
    "reward": reward,
    "success": success,
    "pre_commit": pre_commit,
    "train_command": "mamba run -n tdmpc2 python tdmpc2/train.py task=dog-run model_size=5 steps=50000 seed=1 exp_name=iter3_original_m5_steps50000 enable_wandb=false save_video=false compile=false",
    "eval_command": "mamba run -n tdmpc2 python tdmpc2/evaluate.py task=dog-run model_size=5 checkpoint=/workspace/tdmpc2-codex/logs/dog-run/1/iter3_original_m5_steps50000/models/final.pt eval_episodes=1 save_video=false",
    "checkpoint": "logs/dog-run/1/iter3_original_m5_steps50000/models/final.pt",
    "protocol_tier": "matched_smoke_extended",
    "decision": decision,
    "timestamp": now,
    "comparison_target": {
        "idea_id": "IDEA-004",
        "model_size": 1,
        "steps": 50000,
        "reward": abstraction_reward
    },
    "notes": "Compute-matched original TD-MPC2 comparator for the model_size=1 abstraction/bottleneck probe."
}
with (repo / "scores.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")

state = {
    "updated_at": now,
    "mode": "background_comparator",
    "paper_id": "tdmpc2-codex-dog-run",
    "phase": "matched_comparator_completed",
    "iteration": 3,
    "reward": reward,
    "decision": decision,
    "checkpoint": record["checkpoint"],
    "logs": [
        "logs/autosota/iter3_original_m5_steps50000.log",
        "logs/autosota/iter3_original_m5_steps50000_eval.log"
    ]
}
state_dir = repo / ".autosota_scheduler_test"
state_dir.mkdir(exist_ok=True)
(state_dir / "iter3_original_m5_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

git add scores.jsonl .autosota_scheduler_test/iter3_original_m5_state.json
git commit -m "record autosota iteration 3 matched original comparator"

python - <<'PY'
import json
import subprocess
from pathlib import Path
repo = Path("/workspace/tdmpc2-codex")
last = json.loads((repo / "scores.jsonl").read_text(encoding="utf-8").strip().splitlines()[-1])
if last["reward"] > 55.2:
    subprocess.check_call(["git", "tag", "-f", "_best", "HEAD"], cwd=repo)
PY
