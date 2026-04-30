#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/tdmpc2-codex
ENV_NAME=tdmpc2
MAMBA_BIN=/opt/miniforge3/condabin/mamba
EXP_NAME=iter8_acrobot_se_m1_steps400000
TASK=acrobot-swingup
MODEL_SIZE=1
STEPS=400000
SEED=1
EVAL_FREQ=100000
EVAL_EPISODES=10
SE_COEF=0.01
WAIT_PID="${1:-626356}"
OFFICIAL_CSV="$REPO/results/tdmpc2/acrobot-swingup.csv"
RESULT_DIR="$REPO/results/tdmpc2-codex"
RESULT_CSV="$RESULT_DIR/acrobot-swingup_se_abstraction.csv"
COMPARE_CSV="$RESULT_DIR/acrobot-swingup_se_abstraction_compare.csv"

cd "$REPO"
mkdir -p logs/autosota .autosota_scheduler_test "$RESULT_DIR"
echo "$$" > "logs/autosota/${EXP_NAME}.pid"

export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_LOG="logs/autosota/${EXP_NAME}.log"
META_FILE="logs/autosota/${EXP_NAME}_meta.txt"
CHECKPOINT="$REPO/logs/${TASK}/${SEED}/${EXP_NAME}/models/final.pt"
LOCAL_EVAL_CSV="$REPO/logs/${TASK}/${SEED}/${EXP_NAME}/eval.csv"
START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PRE_COMMIT="$(git rev-parse HEAD)"

{
  echo "timestamp=${START_TS}"
  echo "pre_commit=${PRE_COMMIT}"
  echo "idea_id=IDEA-009"
  echo "task=${TASK}"
  echo "model_size=${MODEL_SIZE}"
  echo "steps=${STEPS}"
  echo "seed=${SEED}"
  echo "eval_freq=${EVAL_FREQ}"
  echo "eval_episodes=${EVAL_EPISODES}"
  echo "se_coef=${SE_COEF}"
  echo "compile=false"
  echo "wait_pid=${WAIT_PID}"
  echo "official_csv=${OFFICIAL_CSV}"
  echo "result_csv=${RESULT_CSV}"
  echo "compare_csv=${COMPARE_CSV}"
  echo "checkpoint=${CHECKPOINT}"
} > "$META_FILE"

if [ -n "$WAIT_PID" ] && ps -p "$WAIT_PID" >/dev/null 2>&1; then
  echo "waiting_for_pid=${WAIT_PID} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RUN_LOG"
  while ps -p "$WAIT_PID" >/dev/null 2>&1; do
    sleep 300
  done
  echo "wait_done $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"
else
  echo "no_active_wait_pid $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RUN_LOG"
fi

echo "starting_train $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"
set +e
"$MAMBA_BIN" run -n "$ENV_NAME" python tdmpc2/train.py \
  task="$TASK" \
  model_size="$MODEL_SIZE" \
  steps="$STEPS" \
  seed="$SEED" \
  eval_freq="$EVAL_FREQ" \
  eval_episodes="$EVAL_EPISODES" \
  exp_name="$EXP_NAME" \
  se_coef="$SE_COEF" \
  enable_wandb=false \
  save_video=false \
  compile=false 2>&1 | tee -a "$RUN_LOG"
train_status=${PIPESTATUS[0]}
set -e
echo "train_exit=${train_status} $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi

python3 scripts/export_acrobot_compile_result.py \
  --official "$OFFICIAL_CSV" \
  --eval-csv "$LOCAL_EVAL_CSV" \
  --result-csv "$RESULT_CSV" \
  --compare-csv "$COMPARE_CSV" \
  --scores-jsonl scores.jsonl \
  --state-json .autosota_scheduler_test/iter8_acrobot_se_abstraction_state.json \
  --run-log "$RUN_LOG" \
  --checkpoint "logs/${TASK}/${SEED}/${EXP_NAME}/models/final.pt" \
  --task "$TASK" \
  --iteration 8 \
  --mode structural_entropy_abstraction \
  --idea-id IDEA-009 \
  --notes "Acrobot-swingup structural-entropy abstraction probe over soft latent transition flow." \
  --seed "$SEED" \
  --model-size "$MODEL_SIZE" \
  --steps "$STEPS" \
  --eval-episodes "$EVAL_EPISODES" \
  --pre-commit "$PRE_COMMIT"
