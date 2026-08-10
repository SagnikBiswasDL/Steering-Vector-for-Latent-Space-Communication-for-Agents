#!/usr/bin/env bash
# Aggressive overnight follow-on after compress_confirm finishes.
# Waits on STOP_POD / chain process, then fills the compression narrative gaps.
set -uo pipefail
cd /workspace/latentmas-baseline || exit 2
# shellcheck disable=SC1091
source env_native.sh

PREV=${PREV:-artifacts/exp2x2/compress_confirm}
OUT=${OUT:-artifacts/exp2x2/night}
VEC=${VEC:-artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt}
POD_ID="${POD_ID:-d7mbbr2zkplgu1}"
mkdir -p "$OUT"
SENTINEL="$OUT/STOP_POD"
rm -f "$SENTINEL"

echo "=== night_aggressive started $(date) ===" | tee -a "$OUT/chain.log"

# Wait for compress_confirm to finish (tmux 'comp' or its STOP_POD)
echo "[night] waiting for compress_confirm..." | tee -a "$OUT/chain.log"
while true; do
  if [[ -f "$PREV/STOP_POD" ]]; then
    echo "[night] found $PREV/STOP_POD" | tee -a "$OUT/chain.log"
    break
  fi
  if ! pgrep -f "run_compress_confirm.sh|exp_2x2_relay_asc.py" >/dev/null 2>&1; then
    # give a moment in case between jobs
    sleep 30
    if ! pgrep -f "run_compress_confirm.sh|exp_2x2_relay_asc.py" >/dev/null 2>&1 \
       && ! tmux has-session -t comp 2>/dev/null; then
      echo "[night] compress chain processes gone; proceeding" | tee -a "$OUT/chain.log"
      break
    fi
  fi
  sleep 120
done

run () {
  local tag="$1"; shift
  echo "[night] START $tag $(date)" | tee -a "$OUT/chain.log"
  python scripts/exp_2x2_relay_asc.py "$@" --out_dir "$OUT/$tag" \
    || echo "[night] FAIL $tag" | tee -a "$OUT/chain.log"
  echo "[night] DONE $tag $(date)" | tee -a "$OUT/chain.log"
}

# --- AIME sanity via Gate1 harness (full LatentMASMethod, not cache-swap) ---
echo "[night] START aime_gate1_sanity $(date)" | tee -a "$OUT/chain.log"
python scripts/gate1_k_curve.py \
  --model_name Qwen/Qwen3-14B --task aime2024 \
  --k_grid 0,10 --max_samples 30 --max_new_tokens 8192 --temperature 0.0 --seed 42 \
  --out_dir "$OUT/aime_gate1_k0k10" \
  || echo "[night] FAIL aime_gate1_sanity" | tee -a "$OUT/chain.log"
echo "[night] DONE aime_gate1_sanity $(date)" | tee -a "$OUT/chain.log"

# --- MedQA compression ladder (find the free regime) ---
for B in 128 64 32; do
  run "medqa_b${B}_s42" \
    --model_name Qwen/Qwen3-14B --task medqa --split test \
    --k 40 --n 60 --batch_size 12 --judger_budget 1024 \
    --relay_budget "$B" --sink 4 --rank 8 --importance key_norm \
    --with_evict --with_shuffled --with_none \
    --asc_vector "$VEC" --asc_coef 40 --seed 42
done

# --- MATH even gentler (b128) ---
run math_b128_s42 \
  --model_name Qwen/Qwen3-14B --task math --split test \
  --k 40 --n 60 --batch_size 12 --judger_budget 2048 \
  --relay_budget 128 --sink 4 --rank 8 --importance key_norm \
  --with_evict --with_shuffled --with_none \
  --asc_vector "$VEC" --asc_coef 40 --seed 42

# --- GSM8K aggressive replicate seed 7 ---
run gsm8k_b16_s7 \
  --model_name Qwen/Qwen3-14B --task gsm8k --split test \
  --k 40 --n 60 --batch_size 20 --judger_budget 768 \
  --relay_budget 16 --sink 4 --rank 8 --importance key_norm \
  --with_evict --with_shuffled \
  --asc_vector "$VEC" --asc_coef 40 --seed 7

# --- Cross-task: MedQA reader with GSM8K donor ---
run medqa_x_gsm8k_s42 \
  --model_name Qwen/Qwen3-14B --task medqa --split test \
  --k 40 --n 60 --batch_size 12 --judger_budget 1024 \
  --relay_budget 64 --sink 4 --rank 8 --importance key_norm \
  --with_shuffled --with_crosstask --with_none \
  --donor_task gsm8k \
  --asc_vector "$VEC" --asc_coef 40 --seed 42

echo "=== NIGHT_AGGRESSIVE_DONE $(date) — STOP THE POD ===" | tee -a "$OUT/chain.log"
date > "$SENTINEL"
runpodctl stop pod "$POD_ID" 2>&1 | tee -a "$OUT/chain.log" \
  || echo "[night] runpodctl stop failed — STOP MANUALLY" | tee -a "$OUT/chain.log"
