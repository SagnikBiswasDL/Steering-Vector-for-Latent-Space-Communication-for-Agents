#!/usr/bin/env bash
# Unattended scaffold chain (runs on the pod in tmux; survives disconnects):
#   waits for the hardening run (tmux session 'synth2') to finish, then runs the
#   shrink grid + multi-seed robustness on 14B MedQA (budget 1024).
#
# Launch:  tmux new-session -d -s scafchain 'bash scripts/run_scaffold_chain.sh > /workspace/scaffold_chain.log 2>&1'
set -uo pipefail

cd /workspace/latentmas-baseline
# shellcheck disable=SC1091
source env_native.sh

echo "=== scaffold chain started $(date) ==="
echo "[chain] waiting for hardening tmux session 'synth2' to finish..."
sleep 5
while tmux has-session -t synth2 2>/dev/null; do sleep 30; done
echo "[chain] hardening finished at $(date)"

# --- Grid A: multi-seed robustness + length sweep ---
echo "[chain] GRID A (robustness + length) $(date)"
python scripts/diag_scaffold_sweep.py --model_name Qwen/Qwen3-14B --task medqa \
  --split test --n 40 --budgets 1024 \
  --variants real,none,synthglobal,synthglobal_s2,synthglobal_s3,synth_l256,synth_l64,synth_l32 \
  --stat_n 8 --stat_split train --out_dir artifacts/diag/14b/grid_len \
  || echo "[chain] grid A exited non-zero"
echo "[chain] grid A done $(date)"

# --- Grid B: rank sweep (+ a short low-rank combo) ---
echo "[chain] GRID B (rank) $(date)"
python scripts/diag_scaffold_sweep.py --model_name Qwen/Qwen3-14B --task medqa \
  --split test --n 40 --budgets 1024 \
  --variants real,none,synth_r64,synth_r32,synth_r16,synth_r8,synth_l64_r16 \
  --stat_n 8 --stat_split train --out_dir artifacts/diag/14b/grid_rank \
  || echo "[chain] grid B exited non-zero"
echo "[chain] grid B done $(date)"

echo "=== scaffold chain complete $(date) ==="
