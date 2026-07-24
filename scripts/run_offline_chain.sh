#!/usr/bin/env bash
# Unattended chain for the offline window:
#   1) wait for the in-flight 4B run to finish
#   2) stability-fixed 4B retrain (reuse mined pairs; grad-clip default + lower LR)
#   3) 14B headline run (fresh mine -> train -> dev go/no-go -> gated heldout)
#
# Launch inside tmux so it is session-independent:
#   tmux new-session -d -s chain 'bash scripts/run_offline_chain.sh'
#
# NOTE: does not use `set -e` at the top level so one failed stage does not abort
# the rest of the chain.
set -uo pipefail

cd /workspace/latentmas-baseline
# shellcheck disable=SC1091
source env_native.sh

echo "=== offline chain started $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- 1) wait for the current 4B driver to finish ---
echo "[chain] waiting for in-flight 'run_boost_experiment.sh 4b' to finish..."
# Give it a moment in case it is mid-startup.
sleep 5
while pgrep -f "run_boost_experiment.sh 4b" >/dev/null 2>&1; do
  sleep 30
done
echo "[chain] in-flight 4B run finished at $(date)"
echo "[chain] --- current 4B dev report (if any) ---"
cat artifacts/ces/boost_4b/dev_k10/report.json 2>/dev/null || echo "[chain] (no dev report)"

# --- 2) stability-fixed 4B retrain, reusing the already-mined pairs ---
TUNED=artifacts/ces/boost_4b_tuned
mkdir -p "$TUNED/pairs_k10"
if cp -f artifacts/ces/boost_4b/pairs_k10/pairs.json "$TUNED/pairs_k10/pairs.json" 2>/dev/null; then
  echo "[chain] reused mined pairs for tuned 4B"
  echo "[chain] starting tuned 4B (grad-clip default, LR=5e-3, 400 steps) at $(date)"
  OUT_ROOT="$TUNED" SKIP_MINE=1 LR=5e-3 STEPS=400 MAX_PAIRS=0 \
    bash scripts/run_boost_experiment.sh 4b || echo "[chain] tuned 4B exited non-zero"
else
  echo "[chain] WARN: could not find mined pairs; re-mining for tuned 4B"
  OUT_ROOT="$TUNED" LR=5e-3 STEPS=400 MAX_PAIRS=0 \
    bash scripts/run_boost_experiment.sh 4b || echo "[chain] tuned 4B exited non-zero"
fi
echo "[chain] tuned 4B stage done at $(date)"
cat "$TUNED/dev_k10/report.json" 2>/dev/null || echo "[chain] (no tuned dev report)"
cat "$TUNED/heldout_k10/report.json" 2>/dev/null || echo "[chain] (no tuned heldout report)"

# --- 3) 14B headline (fresh mine + train + dev go/no-go + gated heldout) ---
echo "[chain] starting 14B headline (grad-clip default, LR=5e-3) at $(date)"
LR=5e-3 bash scripts/run_boost_experiment.sh 14b || echo "[chain] 14B exited non-zero"
echo "[chain] 14B stage done at $(date)"
cat artifacts/ces/boost_14b/dev_k10/report.json 2>/dev/null || echo "[chain] (no 14B dev report)"
cat artifacts/ces/boost_14b/heldout_k10/report.json 2>/dev/null || echo "[chain] (no 14B heldout report)"

echo "=== offline chain complete $(date) ==="
