#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_SILENT=true

if [[ -f /workspace/.wandb_env ]]; then
  set -a
  source /workspace/.wandb_env
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
REPO="${REPO:-/workspace/tdmpc2-codex}"
PROJECT="${WANDB_PROJECT:-tdmpc2-codex}"

cd "$REPO"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  "$PYTHON_BIN" -m wandb login --relogin "$WANDB_API_KEY" >/dev/null
fi

mkdir -p logs/autosota results/tdmpc2-codex .autosota_scheduler_test

run_one() {
  local coef="$1"
  local tag="$2"
  local exp="vastai_coeff_sanity_acrobot_se${tag}_m1_steps100000"
  local run_dir="logs/acrobot-swingup/1/${exp}"
  local run_log="logs/autosota/${exp}.log"
  local result_csv="results/tdmpc2-codex/acrobot_coeff_sanity_${tag}.csv"
  local compare_csv="results/tdmpc2-codex/acrobot_coeff_sanity_${tag}_compare.csv"
  local state_json=".autosota_scheduler_test/${exp}_state.json"
  local summary_json="results/tdmpc2-codex/${exp}_wandb_summary.json"
  local start_ts end_ts status elapsed

  start_ts=$(date +%s)
  {
    echo "starting_coeff_sanity $(date -Is) coef=${coef} exp=${exp}"
    set +e
    "$PYTHON_BIN" tdmpc2/train.py \
      task=acrobot-swingup \
      model_size=1 \
      steps=100000 \
      seed=1 \
      eval_freq=50000 \
      eval_episodes=10 \
      exp_name="$exp" \
      se_coef="$coef" \
      enable_wandb=true \
      wandb_project="$PROJECT" \
      wandb_entity=null \
      wandb_silent=true \
      save_video=false \
      save_agent=true \
      compile=false
    status=$?
    set -e
    end_ts=$(date +%s)
    elapsed=$((end_ts-start_ts))
    echo "train_exit=${status} elapsed_seconds=${elapsed} $(date -Is)"
    if [[ "$status" -ne 0 ]]; then
      exit "$status"
    fi

    "$PYTHON_BIN" scripts/export_acrobot_compile_result.py \
      --official results/tdmpc2/acrobot-swingup.csv \
      --eval-csv "${run_dir}/eval.csv" \
      --result-csv "$result_csv" \
      --compare-csv "$compare_csv" \
      --scores-jsonl scores.jsonl \
      --state-json "$state_json" \
      --run-log "$run_log" \
      --checkpoint "${run_dir}/models/final.pt" \
      --task acrobot-swingup \
      --iteration 10 \
      --mode coefficient_sanity \
      --idea-id IDEA-009 \
      --notes "Tiny coefficient sanity check for lambda_SE=${coef}; official TD-MPC2 CSV is used as reference baseline." \
      --seed 1 \
      --model-size 1 \
      --steps 100000 \
      --eval-episodes 10 \
      --pre-commit "$(git rev-parse HEAD 2>/dev/null || echo payload)"

    "$PYTHON_BIN" - <<PY
from pathlib import Path
import csv
import json

coef = float("$coef")
elapsed = int("$elapsed")
summary = {
    "lambda_se": coef,
    "steps": 100000,
    "seed": 1,
    "gpu": "RTX 5060 Ti",
    "runtime_seconds": elapsed,
    "runtime_hours": elapsed / 3600,
    "baseline_source": "official TD-MPC2 CSV: results/tdmpc2/acrobot-swingup.csv",
}
state_path = Path("$state_json")
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
    for key in ("reward", "official_mean_reward_at_final_step", "train_time", "decision"):
        if key in state:
            summary[key] = state[key]
compare_path = Path("$compare_csv")
if compare_path.exists():
    with compare_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows:
        last = rows[-1]
        summary["final_step"] = int(float(last["step"]))
        summary["candidate_reward"] = float(last["abstraction_compile_reward"])
        summary["official_reference_reward"] = float(last["official_tdmpc2_mean_reward"])
        summary["delta_vs_official"] = float(last["delta"])
Path("$summary_json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    "$PYTHON_BIN" scripts/wandb_attach_artifacts.py \
      --project "$PROJECT" \
      --run-dir "$run_dir" \
      --name "$exp" \
      --summary-json "$summary_json" \
      --artifact "$result_csv" \
      --artifact "$compare_csv" \
      --artifact "$state_json" \
      --artifact "${run_dir}/eval.csv" \
      --artifact "$summary_json"
  } > "$run_log" 2>&1
}

run_one 0.003 0003 &
pid_a=$!
run_one 0.03 003 &
pid_b=$!

status_a=0
status_b=0
wait "$pid_a" || status_a=$?
wait "$pid_b" || status_b=$?

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json

summary = {
    "baseline_source": "official TD-MPC2 CSV used for lambda=0 reference",
    "note": "Summary is local only. Per-run artifacts are attached to the real training W&B runs.",
}
for path in sorted(Path(".autosota_scheduler_test").glob("vastai_coeff_sanity_*_state.json")):
    key = path.stem.removesuffix("_state")
    try:
        summary[key] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        summary[key] = {"error": "invalid_json"}
Path("results/tdmpc2-codex/acrobot_coeff_sanity_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

exit $((status_a || status_b))
