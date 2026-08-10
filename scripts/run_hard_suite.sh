#!/usr/bin/env bash
# Hard-problem suite for the paper thesis:
#   1) MATH + MedQA donor  — does a med cache serve a math problem?
#   2) AIME 2024           — does real≈shuf / compress-free hold when Judger can't re-solve?
#
# Arms: A/B/C/D/E/F + S (shuf) + X (crosstask) + N (none)
# Launch: tmux new -d -s hard 'bash scripts/run_hard_suite.sh > /workspace/hard_suite.log 2>&1'
set -uo pipefail
cd /workspace/latentmas-baseline || exit 2
# shellcheck disable=SC1091
source env_native.sh

OUT=${OUT:-artifacts/exp2x2/hard}
VEC=${VEC:-artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt}
mkdir -p "$OUT"
SENTINEL="$OUT/STOP_POD"
rm -f "$SENTINEL"

echo "=== hard_suite started $(date) ===" | tee -a "$OUT/chain.log"
nvidia-smi -L | tee -a "$OUT/chain.log"

# --- 1) MATH: med→math crosstask + none + compress ---
echo "[hard] MATH n=60 donor=medqa $(date)" | tee -a "$OUT/chain.log"
python scripts/exp_2x2_relay_asc.py \
  --model_name Qwen/Qwen3-14B --task math --split test \
  --k 40 --n 60 --batch_size 12 --judger_budget 2048 \
  --relay_budget 16 --sink 4 --rank 8 --importance key_norm \
  --with_evict --with_shuffled --with_crosstask --with_none \
  --donor_task medqa \
  --asc_vector "$VEC" --asc_coef 40 --seed 42 \
  --out_dir "$OUT/math_meddonor_s42" \
  || echo "[hard] MATH exited non-zero" | tee -a "$OUT/chain.log"
echo "[hard] MATH done $(date)" | tee -a "$OUT/chain.log"

# --- 2) AIME 2024: the paper's hard crux ---
echo "[hard] AIME2024 n=30 $(date)" | tee -a "$OUT/chain.log"
python scripts/exp_2x2_relay_asc.py \
  --model_name Qwen/Qwen3-14B --task aime2024 --split train \
  --k 40 --n 30 --batch_size 4 --judger_budget 8192 \
  --relay_budget 16 --sink 4 --rank 8 --importance key_norm \
  --with_evict --with_shuffled --with_crosstask --with_none \
  --donor_task medqa \
  --asc_vector "$VEC" --asc_coef 40 --seed 42 \
  --out_dir "$OUT/aime2024_meddonor_s42" \
  || echo "[hard] AIME exited non-zero" | tee -a "$OUT/chain.log"
echo "[hard] AIME done $(date)" | tee -a "$OUT/chain.log"

echo "=== HARD_SUITE_DONE $(date) — POD CAN BE STOPPED ===" | tee -a "$OUT/chain.log"
date > "$SENTINEL"
POD_ID="${POD_ID:-1a7b2e145fjhab}"
runpodctl stop pod "$POD_ID" 2>&1 | tee -a "$OUT/chain.log" \
  || echo "[hard] runpodctl stop failed — STOP MANUALLY" | tee -a "$OUT/chain.log"
