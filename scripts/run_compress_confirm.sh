#!/usr/bin/env bash
# Compression-narrative confirm chain (one H200, sequential).
# P0 AIME@K=10 -> P1 MedQA 2x2 -> P1 MATH gentle sweep -> P2 GSM8K seed123
# Launch: tmux new -d -s comp 'bash scripts/run_compress_confirm.sh > /workspace/compress_confirm.log 2>&1'
set -uo pipefail
cd /workspace/latentmas-baseline || exit 2
# shellcheck disable=SC1091
source env_native.sh

OUT=${OUT:-artifacts/exp2x2/compress_confirm}
VEC=${VEC:-artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt}
POD_ID="${POD_ID:-d7mbbr2zkplgu1}"
mkdir -p "$OUT"
SENTINEL="$OUT/STOP_POD"
rm -f "$SENTINEL"

echo "=== compress_confirm started $(date) ===" | tee -a "$OUT/chain.log"
nvidia-smi -L | tee -a "$OUT/chain.log"

run () {
  local tag="$1"; shift
  echo "[comp] START $tag $(date)" | tee -a "$OUT/chain.log"
  python scripts/exp_2x2_relay_asc.py "$@" --out_dir "$OUT/$tag" \
    || echo "[comp] FAIL $tag" | tee -a "$OUT/chain.log"
  echo "[comp] DONE $tag $(date)" | tee -a "$OUT/chain.log"
}

# --- P0: AIME at Gate-1 best K=10 ---
run aime_k10_s42 \
  --model_name Qwen/Qwen3-14B --task aime2024 --split train \
  --k 10 --n 30 --batch_size 4 --judger_budget 8192 \
  --relay_budget 16 --sink 4 --rank 8 --importance key_norm \
  --with_evict --with_shuffled --with_crosstask --with_none \
  --donor_task medqa \
  --asc_vector "$VEC" --asc_coef 40 --seed 42

# --- P1a: MedQA compression 2x2 (aggressive) ---
run medqa_b16_s42 \
  --model_name Qwen/Qwen3-14B --task medqa --split test \
  --k 40 --n 60 --batch_size 12 --judger_budget 1024 \
  --relay_budget 16 --sink 4 --rank 8 --importance key_norm \
  --with_evict --with_shuffled --with_none \
  --asc_vector "$VEC" --asc_coef 40 --seed 42

# --- P1b: MATH gentle compression (budget 64 kept positions) ---
run math_b64_s42 \
  --model_name Qwen/Qwen3-14B --task math --split test \
  --k 40 --n 60 --batch_size 12 --judger_budget 2048 \
  --relay_budget 64 --sink 4 --rank 8 --importance key_norm \
  --with_evict --with_shuffled --with_none \
  --asc_vector "$VEC" --asc_coef 40 --seed 42

# --- P1c: MATH medium compression (budget 32) ---
run math_b32_s42 \
  --model_name Qwen/Qwen3-14B --task math --split test \
  --k 40 --n 60 --batch_size 12 --judger_budget 2048 \
  --relay_budget 32 --sink 4 --rank 8 --importance key_norm \
  --with_evict --with_shuffled \
  --asc_vector "$VEC" --asc_coef 40 --seed 42

# --- P2: GSM8K aggressive second seed ---
run gsm8k_b16_s123 \
  --model_name Qwen/Qwen3-14B --task gsm8k --split test \
  --k 40 --n 60 --batch_size 20 --judger_budget 768 \
  --relay_budget 16 --sink 4 --rank 8 --importance key_norm \
  --with_evict --with_shuffled \
  --asc_vector "$VEC" --asc_coef 40 --seed 123

echo "=== COMPRESS_CONFIRM_DONE $(date) — STOP THE POD ===" | tee -a "$OUT/chain.log"
date > "$SENTINEL"
runpodctl stop pod "$POD_ID" 2>&1 | tee -a "$OUT/chain.log" \
  || echo "[comp] runpodctl stop failed — STOP MANUALLY" | tee -a "$OUT/chain.log"
