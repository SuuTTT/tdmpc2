#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/tdmpc2-codex
ENV_NAME=tdmpc2
MAMBA_BIN=/opt/miniforge3/condabin/mamba
EXP_NAME=iter5_fastplanner_m1_steps50000
TASK=dog-run
MODEL_SIZE=1
STEPS=50000
SEED=1
EVAL_EPISODES=10
BASELINE_REWARD=50.5

cd "$REPO"
mkdir -p logs/autosota .autosota_scheduler_test
echo "$$" > "logs/autosota/${EXP_NAME}.pid"

export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PRE_COMMIT="$(git rev-parse HEAD)"
RUN_LOG="logs/autosota/${EXP_NAME}.log"
EVAL_LOG="logs/autosota/${EXP_NAME}_eval10.log"
CHECKPOINT="$REPO/logs/${TASK}/${SEED}/${EXP_NAME}/models/final.pt"

{
  echo "timestamp=${START_TS}"
  echo "pre_commit=${PRE_COMMIT}"
  echo "idea_id=IDEA-007"
  echo "task=${TASK}"
  echo "model_size=${MODEL_SIZE}"
  echo "steps=${STEPS}"
  echo "num_samples=256"
  echo "num_elites=32"
  echo "num_pi_trajs=12"
  echo "iterations=4"
  echo "checkpoint=${CHECKPOINT}"
} > "logs/autosota/${EXP_NAME}_meta.txt"

echo "starting_train $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RUN_LOG"
set +e
"$MAMBA_BIN" run -n "$ENV_NAME" python tdmpc2/train.py \
  task="$TASK" \
  model_size="$MODEL_SIZE" \
  steps="$STEPS" \
  seed="$SEED" \
  exp_name="$EXP_NAME" \
  num_samples=256 \
  num_elites=32 \
  num_pi_trajs=12 \
  iterations=4 \
  enable_wandb=false \
  save_video=false \
  compile=false 2>&1 | tee -a "$RUN_LOG"
train_status=${PIPESTATUS[0]}
set -e
echo "train_exit=${train_status} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

echo "starting_eval $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$EVAL_LOG"
set +e
"$MAMBA_BIN" run -n "$ENV_NAME" python tdmpc2/evaluate.py \
  task="$TASK" \
  model_size="$MODEL_SIZE" \
  checkpoint="$CHECKPOINT" \
  eval_episodes="$EVAL_EPISODES" \
  save_video=false \
  compile=false 2>&1 | tee -a "$EVAL_LOG"
eval_status=${PIPESTATUS[0]}
set -e
echo "eval_exit=${eval_status} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$EVAL_LOG"
if [ "$eval_status" -ne 0 ]; then
  exit "$eval_status"
fi

python3 - <<'PY'
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo = Path("/workspace/tdmpc2-codex")
run_log = repo / "logs/autosota/iter5_fastplanner_m1_steps50000.log"
eval_log = repo / "logs/autosota/iter5_fastplanner_m1_steps50000_eval10.log"

def parse_reward(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"R:\s*([-+]?\d+(?:\.\d+)?)\s+S:\s*([-+]?\d+(?:\.\d+)?)", text)
    if not matches:
        raise SystemExit(f"Could not parse reward from {path}")
    return tuple(map(float, matches[-1]))

def parse_final_train_time(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"T:\s*([0-9]+:[0-9]{2}:[0-9]{2})", text)
    return matches[-1] if matches else None

reward, success = parse_reward(eval_log)
train_time = parse_final_train_time(run_log)
baseline_reward = 50.5
decision = "FASTPLANNER_COMPETITIVE" if reward >= baseline_reward * 0.9 else "FASTPLANNER_TOO_DEGRADED"
now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
pre_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

record = {
    "iteration": 5,
    "mode": "speed_optimization",
    "idea_id": "IDEA-007",
    "granularity": "PARAM",
    "task": "dog-run",
    "metric": "episode_reward",
    "reward": reward,
    "success": success,
    "train_time": train_time,
    "eval_episodes": 10,
    "pre_commit": pre_commit,
    "protocol_tier": "matched_smoke_extended_eval10",
    "decision": decision,
    "timestamp": now,
    "config": {
        "model_size": 1,
        "steps": 50000,
        "num_samples": 256,
        "num_elites": 32,
        "num_pi_trajs": 12,
        "iterations": 4
    },
    "comparison_target": {
        "idea_id": "IDEA-004",
        "model_size": 1,
        "steps": 50000,
        "eval_episodes": 10,
        "reward": baseline_reward
    },
    "checkpoint": "logs/dog-run/1/iter5_fastplanner_m1_steps50000/models/final.pt",
    "notes": "Planner-cost reduction speed probe for the validated model_size=1 abstraction path."
}
with (repo / "scores.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")

idea_path = repo / "idea_library.md"
idea = idea_path.read_text(encoding="utf-8")
result = f"Completed with eval10 reward {reward:.1f}; train time {train_time or 'unknown'}; decision {decision}."
idea = re.sub(
    r"(### IDEA-007: Planner-Cost Reduction for Faster Abstraction Runs.*?- \\*\\*Status\\*\\*: )IN-PROGRESS(\\n- \\*\\*Result\\*\\*: ).*?(\\n\\Z|\\n\\n### )",
    lambda m: m.group(1) + ("SUCCESS" if decision == "FASTPLANNER_COMPETITIVE" else "FAILED") + m.group(2) + result + m.group(3),
    idea,
    flags=re.S,
)
idea_path.write_text(idea, encoding="utf-8")

state = {
    "updated_at": now,
    "mode": "speed_optimization",
    "paper_id": "tdmpc2-codex-dog-run",
    "phase": "speed_probe_completed",
    "iteration": 5,
    "decision": decision,
    "reward": reward,
    "train_time": train_time,
    "logs": [
        "logs/autosota/iter5_fastplanner_m1_steps50000.log",
        "logs/autosota/iter5_fastplanner_m1_steps50000_eval10.log"
    ]
}
(repo / ".autosota_scheduler_test/iter5_fastplanner_state.json").write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

git add scores.jsonl idea_library.md .autosota_scheduler_test/iter5_fastplanner_state.json
git commit -m "record autosota iteration 5 fast planner result"
