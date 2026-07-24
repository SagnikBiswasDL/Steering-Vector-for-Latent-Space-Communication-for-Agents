#!/usr/bin/env bash
# All-token latent steering variant (apply_to=all): the decisive test of whether
# steering EVERY latent token position (not just the last) gives the intervention
# any causal leverage over the Judger. Reuses the already-mined 4B pairs.
#   train (apply_to=all) -> coef sweep (coefs 1,2,4,8,-8) on dev.
set -uo pipefail

cd /workspace/latentmas-baseline
# shellcheck disable=SC1091
source env_native.sh

MODEL="${MODEL:-Qwen/Qwen3-4B}"
LAYER="${LAYER:-20}"
OUT="${OUT:-artifacts/ces/boost_4b_all}"
PAIRS="${PAIRS:-artifacts/ces/boost_4b/pairs_k10/pairs.json}"
mkdir -p "$OUT/train"

echo "=== all-tokens TRAIN $(date) (model=$MODEL layer=$LAYER) ==="
if [[ ! -f "$PAIRS" ]]; then echo "FATAL: pairs not found at $PAIRS"; exit 1; fi
python scripts/train_ces_claim_b.py \
  --model_name "$MODEL" --pairs "$PAIRS" --objective ces_rank \
  --k_low 10 --ces_layer "$LAYER" --ces_coef 1.0 --apply_to all \
  --max_pairs 0 --steps 400 --lr 5e-3 --max_new_tokens 4096 \
  --out_dir "$OUT/train" || { echo "FATAL: train failed"; exit 1; }

echo "=== all-tokens COEF SWEEP $(date) ==="
python scripts/coef_sweep.py \
  --model_name "$MODEL" --ces_vector "$OUT/train/ces_latest.pt" \
  --k 10 --coefs 1,2,4,8,-8 --split dev --max_samples 40 \
  --max_new_tokens 2048 --out_dir "$OUT/coef_sweep" || { echo "FATAL: sweep failed"; exit 1; }

echo "=== all-tokens DONE $(date) ==="
