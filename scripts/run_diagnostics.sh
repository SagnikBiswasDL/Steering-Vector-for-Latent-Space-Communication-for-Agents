#!/usr/bin/env bash
# "Steer the reader" diagnostics (plan GPU-0 branch):
#   1) cache-usage: token-cap x {real,shuffled,zero,none}  -> does the Judger USE the cache?
#   2) answer-probes: is the gold answer decodable from upstream latents (linear vs MLP)?
#
# Usage: scripts/run_diagnostics.sh {4b|14b}
# Env overrides: N_CACHE, BUDGETS, N_PROBE, STRIDE, K.
set -uo pipefail

SIZE="${1:-4b}"
case "$SIZE" in
  4b)  MODEL="${MODEL:-Qwen/Qwen3-4B}" ;;
  14b) MODEL="${MODEL:-Qwen/Qwen3-14B}" ;;
  *) echo "usage: $0 {4b|14b}" >&2; exit 2 ;;
esac

cd /workspace/latentmas-baseline
# shellcheck disable=SC1091
source env_native.sh

K="${K:-10}"
OUT="artifacts/diag/${SIZE}"
mkdir -p "$OUT"

echo "=== [1/2] cache-usage ($MODEL) $(date) ==="
python scripts/diag_cache_usage.py \
  --model_name "$MODEL" --k "$K" --split test \
  --n "${N_CACHE:-60}" --budgets "${BUDGETS:-128,256,512,1024}" \
  --conditions real,shuffled,zero,none \
  --out_dir "$OUT/cache_usage" || echo "[diag] cache_usage exited non-zero"

echo "=== [2/2] answer-probes ($MODEL) $(date) ==="
python scripts/diag_answer_probes.py \
  --model_name "$MODEL" --k "$K" --split train \
  --n "${N_PROBE:-200}" --layer_stride "${STRIDE:-2}" \
  --out_dir "$OUT/answer_probes" || echo "[diag] answer_probes exited non-zero"

echo "=== diagnostics done $(date) ==="
echo "--- cache_usage report ---"; cat "$OUT/cache_usage/report.json" 2>/dev/null | tail -40 || true
echo "--- answer_probes best-per-role ---"; python3 -c "import json;r=json.load(open('$OUT/answer_probes/report.json'));print(json.dumps(r.get('best_per_role',{}),indent=2))" 2>/dev/null || true
