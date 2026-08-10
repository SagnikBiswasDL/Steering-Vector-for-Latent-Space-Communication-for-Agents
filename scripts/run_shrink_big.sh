#!/usr/bin/env bash
# Bigger-compute shrink gamble (unattended).
#
# Goal: short learned scaffold (m=64/128/256) matching real LatentMAS accuracy
# while cutting KV length (~1023 -> m), plus multi-seed confirmation of fixed short synth.
#
# Launch on pod (direct TCP preferred):
#   tmux new-session -d -s shrink \
#     'bash scripts/run_shrink_big.sh > /workspace/shrink_big.log 2>&1'
# Watch:
#   tail -f /workspace/shrink_big.log
# Stop signal when done:
#   /workspace/latentmas-baseline/artifacts/ces/STOP_POD
set -uo pipefail

ROOT="${ROOT:-/workspace/latentmas-baseline}"
cd "$ROOT"
# shellcheck disable=SC1091
source env_native.sh

OUT_ROOT="${OUT_ROOT:-artifacts/ces/shrink_big}"
mkdir -p "$OUT_ROOT"
SENTINEL="$OUT_ROOT/STOP_POD"
rm -f "$SENTINEL"

echo "=== shrink_big started $(date) ===" | tee -a "$OUT_ROOT/chain.log"
echo "[chain] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1)" | tee -a "$OUT_ROOT/chain.log"

# --- Phase A: multi-seed short fixed synth (noise check on l32/l64/l128) ---
echo "[chain] PHASE A multi-seed length $(date)" | tee -a "$OUT_ROOT/chain.log"
python scripts/diag_scaffold_sweep.py --model_name Qwen/Qwen3-14B --task medqa \
  --split test --n 40 --budgets 1024 \
  --variants real,none,synthglobal,synthglobal_s2,synthglobal_s3,synth_l128,synth_l128_s2,synth_l64,synth_l64_s2,synth_l64_s3,synth_l32,synth_l32_s2 \
  --stat_n 16 --stat_split train --out_dir "$OUT_ROOT/grid_len_multiseed" \
  || echo "[chain] phase A exited non-zero" | tee -a "$OUT_ROOT/chain.log"
echo "[chain] PHASE A done $(date)" | tee -a "$OUT_ROOT/chain.log"

# --- Phase B: learned short scaffolds (stronger obj + init), largest first ---
for M in 256 128 64; do
  echo "[chain] PHASE B train m=$M $(date)" | tee -a "$OUT_ROOT/chain.log"
  python scripts/train_synth_scaffold.py \
    --model_name Qwen/Qwen3-14B --task medqa \
    --m "$M" --init pool --objective nll+kl --alpha_kl 1.0 \
    --n_train 80 --stat_n 16 --steps 800 --lr 3e-2 --min_lr 1e-3 \
    --teacher_max_tok 512 --eval_n 40 --budget 1024 --seed 42 \
    --out_dir "$OUT_ROOT/learned_m${M}" \
    || echo "[chain] m=$M exited non-zero" | tee -a "$OUT_ROOT/chain.log"
  echo "[chain] m=$M done $(date)" | tee -a "$OUT_ROOT/chain.log"
  # if a run already looks like a win (>= real-0.05 and >> none), still finish others
done

# --- Phase C: if best m looks promising, quick GSM8K transfer of that checkpoint is OUT OF SCOPE
#     (eval is baked into train script on MedQA). Optional: re-run best m with --task gsm8k later.

echo "=== ALL SHRINK_BIG RUNS DONE $(date) — POD CAN BE STOPPED ===" | tee -a "$OUT_ROOT/chain.log"
date > "$SENTINEL"
echo "STOP_POD written to $SENTINEL" | tee -a "$OUT_ROOT/chain.log"

# Best-effort auto-stop (needs runpodctl API key); harmless if it fails.
POD_ID="${POD_ID:-1a7b2e145fjhab}"
runpodctl stop pod "$POD_ID" 2>&1 | tee -a "$OUT_ROOT/chain.log" \
  || echo "[chain] runpodctl stop failed — STOP THE POD MANUALLY" | tee -a "$OUT_ROOT/chain.log"
