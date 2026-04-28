#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/tdmpc2-codex
ENV_NAME=tdmpc2
MAMBA_BIN=/opt/miniforge3/condabin/mamba
TASK=dog-run
SEED=1
EVAL_EPISODES=10

cd "$REPO"
mkdir -p logs/autosota .autosota_scheduler_test
echo "$$" > logs/autosota/iter4_validate_abstraction_vs_original.pid

export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ABS_CKPT="$REPO/logs/${TASK}/${SEED}/iter2_abstraction_m1_steps50000/models/final.pt"
ORG_CKPT="$REPO/logs/${TASK}/${SEED}/iter3_original_m5_steps50000/models/final.pt"
ABS_LOG=logs/autosota/iter4_validate_abstraction_m1_eval10.log
ORG_LOG=logs/autosota/iter4_validate_original_m5_eval10.log

echo "starting_abstraction_eval $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$ABS_LOG"
set +e
"$MAMBA_BIN" run -n "$ENV_NAME" python tdmpc2/evaluate.py \
  task="$TASK" \
  model_size=1 \
  checkpoint="$ABS_CKPT" \
  eval_episodes="$EVAL_EPISODES" \
  save_video=false \
  compile=false 2>&1 | tee -a "$ABS_LOG"
abs_status=${PIPESTATUS[0]}
set -e
echo "abstraction_eval_exit=${abs_status} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$ABS_LOG"
if [ "$abs_status" -ne 0 ]; then
  exit "$abs_status"
fi

echo "starting_original_eval $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$ORG_LOG"
set +e
"$MAMBA_BIN" run -n "$ENV_NAME" python tdmpc2/evaluate.py \
  task="$TASK" \
  model_size=5 \
  checkpoint="$ORG_CKPT" \
  eval_episodes="$EVAL_EPISODES" \
  save_video=false \
  compile=false 2>&1 | tee -a "$ORG_LOG"
org_status=${PIPESTATUS[0]}
set -e
echo "original_eval_exit=${org_status} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$ORG_LOG"
if [ "$org_status" -ne 0 ]; then
  exit "$org_status"
fi

python3 - <<'PY'
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo = Path("/workspace/tdmpc2-codex")

def parse_reward(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"R:\s*([-+]?\d+(?:\.\d+)?)\s+S:\s*([-+]?\d+(?:\.\d+)?)", text)
    if not matches:
        raise SystemExit(f"Could not parse reward from {path}")
    return tuple(map(float, matches[-1]))

abs_reward, abs_success = parse_reward(repo / "logs/autosota/iter4_validate_abstraction_m1_eval10.log")
org_reward, org_success = parse_reward(repo / "logs/autosota/iter4_validate_original_m5_eval10.log")
decision = "ABSTRACTION_VALIDATED" if abs_reward > org_reward else "ORIGINAL_VALIDATED"
now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
pre_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

record = {
    "iteration": 4,
    "mode": "robustness_validation",
    "task": "dog-run",
    "metric": "episode_reward",
    "eval_episodes": 10,
    "pre_commit": pre_commit,
    "protocol_tier": "matched_smoke_extended_eval10",
    "decision": decision,
    "timestamp": now,
    "abstraction": {
        "idea_id": "IDEA-004",
        "model_size": 1,
        "steps": 50000,
        "reward": abs_reward,
        "success": abs_success,
        "checkpoint": "logs/dog-run/1/iter2_abstraction_m1_steps50000/models/final.pt"
    },
    "original": {
        "idea_id": "COMPARATOR-M5-50000",
        "model_size": 5,
        "steps": 50000,
        "reward": org_reward,
        "success": org_success,
        "checkpoint": "logs/dog-run/1/iter3_original_m5_steps50000/models/final.pt"
    },
    "notes": "10-episode evaluation of the two 50000-step checkpoints to reduce one-episode noise."
}
with (repo / "scores.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, sort_keys=True) + "\n")

state = {
    "updated_at": now,
    "mode": "robustness_validation",
    "paper_id": "tdmpc2-codex-dog-run",
    "phase": "validation_completed",
    "iteration": 4,
    "decision": decision,
    "abstraction_reward": abs_reward,
    "original_reward": org_reward,
    "logs": [
        "logs/autosota/iter4_validate_abstraction_m1_eval10.log",
        "logs/autosota/iter4_validate_original_m5_eval10.log"
    ]
}
(repo / ".autosota_scheduler_test/iter4_validation_state.json").write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

git add scores.jsonl .autosota_scheduler_test/iter4_validation_state.json
git commit -m "record autosota iteration 4 validation"
