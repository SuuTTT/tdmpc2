#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-35894623}"
SSH_HOST="${SSH_HOST:-62.107.25.198}"
SSH_PORT="${SSH_PORT:-38845}"
REMOTE_PID_FILE="${REMOTE_PID_FILE:-/workspace/manual_rerun.pid}"
REMOTE_LOG="${REMOTE_LOG:-/workspace/manual_rerun.log}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/tdmpc2-codex}"
EXP_NAME="${EXP_NAME:-vastai_rerun_acrobot_se_m1_steps400000}"
REMOTE_ARTIFACT="$REMOTE_REPO/results/tdmpc2-codex/${EXP_NAME}_artifacts.tgz"
LOCAL_DIR="${LOCAL_DIR:-/workspace/tdmpc2-codex/remote_results/${EXP_NAME}}"
COEFF_PID_FILE="${COEFF_PID_FILE:-/workspace/coeff_sanity.pid}"
COEFF_LOG="${COEFF_LOG:-/workspace/coeff_sanity_launcher.log}"
POLL_SECONDS="${POLL_SECONDS:-600}"
MAX_POLLS="${MAX_POLLS:-120}"

SSH_OPTS=(
  -i "$HOME/.ssh/id_ed25519"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/tmp/known_hosts_vast_tdmpc2
  -p "$SSH_PORT"
)
SCP_OPTS=(
  -i "$HOME/.ssh/id_ed25519"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/tmp/known_hosts_vast_tdmpc2
  -P "$SSH_PORT"
)

mkdir -p "$LOCAL_DIR"

for i in $(seq 1 "$MAX_POLLS"); do
  stamp="$(date -Is)"
  status="$(
    ssh "${SSH_OPTS[@]}" "root@$SSH_HOST" \
      "main_running=0; coeff_running=0; test -s '$REMOTE_PID_FILE' && ps -p \$(cat '$REMOTE_PID_FILE') >/dev/null 2>&1 && main_running=1; test -s '$COEFF_PID_FILE' && ps -p \$(cat '$COEFF_PID_FILE') >/dev/null 2>&1 && coeff_running=1; if test -s '$REMOTE_ARTIFACT' && test \$coeff_running -eq 0; then echo artifact; elif test \$main_running -eq 1 || test \$coeff_running -eq 1; then echo running; else echo stopped; fi" \
      2>/dev/null || echo ssh_failed
  )"
  echo "[$stamp] poll=$i status=$status"

  if [[ "$status" == "artifact" ]]; then
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$REMOTE_ARTIFACT" "$LOCAL_DIR/" || true
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$REMOTE_LOG" "$LOCAL_DIR/manual_rerun.log" || true
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$COEFF_LOG" "$LOCAL_DIR/coeff_sanity_launcher.log" || true
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$REMOTE_REPO/logs/acrobot-swingup/1/$EXP_NAME/eval.csv" "$LOCAL_DIR/eval.csv" || true
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$REMOTE_REPO/results/tdmpc2-codex/acrobot_coeff_sanity_0003.csv" "$LOCAL_DIR/" || true
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$REMOTE_REPO/results/tdmpc2-codex/acrobot_coeff_sanity_0003_compare.csv" "$LOCAL_DIR/" || true
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$REMOTE_REPO/results/tdmpc2-codex/acrobot_coeff_sanity_003.csv" "$LOCAL_DIR/" || true
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$REMOTE_REPO/results/tdmpc2-codex/acrobot_coeff_sanity_003_compare.csv" "$LOCAL_DIR/" || true
    vastai destroy instance "$INSTANCE_ID" --raw || true
    echo "[$(date -Is)] artifact copied and instance destroy requested"
    exit 0
  fi

  if [[ "$status" == "stopped" ]]; then
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$REMOTE_LOG" "$LOCAL_DIR/manual_rerun.log" || true
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$COEFF_LOG" "$LOCAL_DIR/coeff_sanity_launcher.log" || true
    scp "${SCP_OPTS[@]}" "root@$SSH_HOST:$REMOTE_REPO/logs/acrobot-swingup/1/$EXP_NAME/eval.csv" "$LOCAL_DIR/eval.csv" || true
    vastai destroy instance "$INSTANCE_ID" --raw || true
    echo "[$(date -Is)] process stopped without artifact; logs copied and instance destroy requested"
    exit 1
  fi

  sleep "$POLL_SECONDS"
done

echo "[$(date -Is)] monitor timed out; leaving instance for manual inspection" >&2
exit 2
