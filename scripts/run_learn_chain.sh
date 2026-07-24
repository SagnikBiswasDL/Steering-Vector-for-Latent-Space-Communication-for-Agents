#!/usr/bin/env bash
# Fill the GPU window: after the running m=64 learned-scaffold job (tmux 'learn')
# finishes, train m=32 and m=16, then signal that the pod can be stopped.
# Auto-stop is attempted via runpodctl (no-op without an API key).
set -uo pipefail
cd /workspace/latentmas-baseline
# shellcheck disable=SC1091
source env_native.sh

POD_ID="${POD_ID:-dhdjbpkrk690q2}"
echo "=== learn chain started $(date) ==="
echo "[chain] waiting for m=64 (tmux 'learn') to finish..."
sleep 5
while tmux has-session -t learn 2>/dev/null; do sleep 60; done
echo "[chain] m=64 finished $(date)"

for M in 32 16; do
  echo "[chain] training m=$M $(date)"
  python scripts/train_synth_scaffold.py --model_name Qwen/Qwen3-14B --task medqa \
    --m "$M" --n_train 40 --stat_n 8 --steps 400 --lr 5e-2 --teacher_max_tok 512 \
    --eval_n 40 --budget 1024 --out_dir "artifacts/ces/synth_scaffold_m${M}" \
    || echo "[chain] m=$M exited non-zero"
  echo "[chain] m=$M done $(date)"
done

echo "=== ALL LEARN RUNS DONE $(date) — POD CAN BE STOPPED ==="
# Best-effort auto-stop (needs runpodctl API key configured); harmless if it fails.
runpodctl stop pod "$POD_ID" 2>&1 || echo "[chain] runpodctl stop failed (no API key) — STOP THE POD MANUALLY"
