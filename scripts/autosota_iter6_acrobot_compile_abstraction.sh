#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/tdmpc2-codex
ENV_NAME=tdmpc2
MAMBA_BIN=/opt/miniforge3/condabin/mamba
EXP_NAME=iter6_acrobot_compile_abstraction_m1_steps4000000
TASK=acrobot-swingup
MODEL_SIZE=1
STEPS=4000000
SEED=1
EVAL_FREQ=100000
EVAL_EPISODES=10
OFFICIAL_CSV="$REPO/results/tdmpc2/acrobot-swingup.csv"
RESULT_DIR="$REPO/results/tdmpc2-codex"
RESULT_CSV="$RESULT_DIR/acrobot-swingup_abstraction_compile.csv"
COMPARE_CSV="$RESULT_DIR/acrobot-swingup_abstraction_compile_compare.csv"

cd "$REPO"
mkdir -p logs/autosota .autosota_scheduler_test "$RESULT_DIR"
echo "$$" > "logs/autosota/${EXP_NAME}.pid"

unset TORCHDYNAMO_DISABLE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PRE_COMMIT="$(git rev-parse HEAD)"
RUN_LOG="logs/autosota/${EXP_NAME}.log"
META_FILE="logs/autosota/${EXP_NAME}_meta.txt"
CHECKPOINT="$REPO/logs/${TASK}/${SEED}/${EXP_NAME}/models/final.pt"
LOCAL_EVAL_CSV="$REPO/logs/${TASK}/${SEED}/${EXP_NAME}/eval.csv"

{
  echo "timestamp=${START_TS}"
  echo "pre_commit=${PRE_COMMIT}"
  echo "idea_id=IDEA-004"
  echo "task=${TASK}"
  echo "candidate_model_size=${MODEL_SIZE}"
  echo "candidate_steps=${STEPS}"
  echo "seed=${SEED}"
  echo "eval_freq=${EVAL_FREQ}"
  echo "eval_episodes=${EVAL_EPISODES}"
  echo "compile=true"
  echo "official_csv=${OFFICIAL_CSV}"
  echo "result_csv=${RESULT_CSV}"
  echo "compare_csv=${COMPARE_CSV}"
  echo "checkpoint=${CHECKPOINT}"
} > "$META_FILE"

echo "starting_train $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$RUN_LOG"
set +e
"$MAMBA_BIN" run -n "$ENV_NAME" python tdmpc2/train.py \
  task="$TASK" \
  model_size="$MODEL_SIZE" \
  steps="$STEPS" \
  seed="$SEED" \
  eval_freq="$EVAL_FREQ" \
  eval_episodes="$EVAL_EPISODES" \
  exp_name="$EXP_NAME" \
  enable_wandb=false \
  save_video=false \
  compile=true 2>&1 | tee -a "$RUN_LOG"
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
  --state-json .autosota_scheduler_test/iter6_acrobot_compile_abstraction_state.json \
  --run-log "$RUN_LOG" \
  --checkpoint "logs/${TASK}/${SEED}/${EXP_NAME}/models/final.pt" \
  --task "$TASK" \
  --seed "$SEED" \
  --model-size "$MODEL_SIZE" \
  --steps "$STEPS" \
  --eval-episodes "$EVAL_EPISODES" \
  --pre-commit "$PRE_COMMIT"

