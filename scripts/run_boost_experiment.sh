#!/usr/bin/env bash
# CES latent-agent "boost at best K" experiment driver.
#
# Runs the plan's Phase A (mine -> train -> dev go/no-go) and, if the go/no-go
# passes, Phase B/C (held-out test eval with causal controls). One command per
# model size. Vectors do NOT transfer across hidden sizes, so 4B and 14B each
# train their own vector.
#
# Usage:
#   scripts/run_boost_experiment.sh 4b            # full Phase A + (gated) held-out eval
#   scripts/run_boost_experiment.sh 14b
#   FORCE_EVAL=1 scripts/run_boost_experiment.sh 4b   # run held-out even on NO-GO
#   SKIP_MINE=1 SKIP_TRAIN=1 scripts/run_boost_experiment.sh 4b  # eval only (reuse artifacts)
#
# Env overrides (with defaults): K, CES_LAYER, LR, STEPS, MAX_PAIRS, COEF,
#   N_TRAIN, N_DEV, N_TEST, MAX_NEW_TOKENS, SEED, OBJECTIVE, OUT_ROOT.
set -euo pipefail

SIZE="${1:-4b}"
case "$SIZE" in
  4b)  MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}";  DEF_LAYER=20 ;;
  14b) MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"; DEF_LAYER=28 ;;
  *) echo "usage: $0 {4b|14b}" >&2; exit 2 ;;
esac

# Operating point = best unsteered K on MedQA (Gate 1: K=10).
K="${K:-10}"
CES_LAYER="${CES_LAYER:-$DEF_LAYER}"
OBJECTIVE="${OBJECTIVE:-ces_rank}"
LR="${LR:-1e-2}"
STEPS="${STEPS:-300}"
MAX_PAIRS="${MAX_PAIRS:-60}"     # 0 = use every mined pair
COEF="${COEF:-1.0}"
APPLY_TO="${APPLY_TO:-last}"      # 'last' = last latent token only; 'all' = every latent token
N_TRAIN="${N_TRAIN:-120}"        # MedQA train split is [100,220) = 120 items
N_DEV="${N_DEV:-80}"             # MedQA dev split is [220:] ~ 80 items
N_TEST="${N_TEST:-100}"          # MedQA test split is [0,100)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
SEED="${SEED:-42}"

OUT_ROOT="${OUT_ROOT:-artifacts/ces/boost_${SIZE}}"
PAIRS_DIR="${OUT_ROOT}/pairs_k${K}"
TRAIN_DIR="${OUT_ROOT}/train_k${K}_L${CES_LAYER}"
DEV_DIR="${OUT_ROOT}/dev_k${K}"
TEST_DIR="${OUT_ROOT}/heldout_k${K}"
VEC="${TRAIN_DIR}/ces_latest.pt"

PY="${PY:-python}"

echo "=== CES boost experiment: ${SIZE} (${MODEL_NAME}) ==="
echo "K=${K} layer=${CES_LAYER} objective=${OBJECTIVE} steps=${STEPS} lr=${LR} coef=${COEF} apply_to=${APPLY_TO} seed=${SEED}"
echo "out=${OUT_ROOT}"
mkdir -p "${OUT_ROOT}"

# --- Phase A.1: mine correctness pairs at K on the TRAIN split ---
if [[ "${SKIP_MINE:-0}" != "1" ]]; then
  echo "--- [mine] single_k=${K} on train split (n=${N_TRAIN}) ---"
  "${PY}" scripts/mine_budget_pairs.py \
    --model_name "${MODEL_NAME}" \
    --single_k "${K}" \
    --split train --max_samples "${N_TRAIN}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --generate_bs 1 --temperature 0.0 --seed "${SEED}" \
    --out_dir "${PAIRS_DIR}"
fi

# --- Phase A.2: train the CES vector (ranking loss) ---
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "--- [train] ${OBJECTIVE} at K=${K}, layer=${CES_LAYER} ---"
  "${PY}" scripts/train_ces_claim_b.py \
    --model_name "${MODEL_NAME}" \
    --pairs "${PAIRS_DIR}/pairs.json" \
    --objective "${OBJECTIVE}" \
    --k_low "${K}" \
    --ces_layer "${CES_LAYER}" \
    --ces_coef "${COEF}" \
    --max_pairs "${MAX_PAIRS}" \
    --steps "${STEPS}" \
    --lr "${LR}" \
    --apply_to "${APPLY_TO}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --out_dir "${TRAIN_DIR}"
fi

# --- Phase A.3: dev go/no-go (boost, alpha-sign control) on the DEV split ---
echo "--- [dev] go/no-go on dev split (n=${N_DEV}) with negate control ---"
"${PY}" scripts/eval_ces_heldout.py \
  --model_name "${MODEL_NAME}" \
  --ces_vector "${VEC}" \
  --mode boost --control negate \
  --split dev \
  --k_low "${K}" --k_full "${K}" \
  --max_samples "${N_DEV}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.0 --seed "${SEED}" \
  --ces_layer "${CES_LAYER}" --ces_coef "${COEF}" \
  --out_dir "${DEV_DIR}"

# Gate: read the dev decision. GO if steered_pos is a credible positive AND the
# negate control is not itself a credible positive (polarity holds).
DECISION="$("${PY}" - "${DEV_DIR}/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
pos = r.get("paired_vs_unsteered", {}).get("steered_pos", {})
neg = r.get("paired_vs_unsteered", {}).get("steered_neg", {})
go = bool(pos.get("credible_positive")) and not bool(neg.get("credible_positive"))
print("GO" if go else "NOGO")
PY
)"
echo "=== dev go/no-go: ${DECISION} ==="

if [[ "${DECISION}" != "GO" && "${FORCE_EVAL:-0}" != "1" ]]; then
  echo "NO-GO on dev. Stopping before held-out eval (set FORCE_EVAL=1 to override)."
  echo "This is a valid negative result: document it (probe predicts weak upstream signal)."
  exit 0
fi

# --- Phase B/C: held-out TEST eval with full causal controls ---
echo "--- [test] held-out eval on test split (n=${N_TEST}) with all controls ---"
"${PY}" scripts/eval_ces_heldout.py \
  --model_name "${MODEL_NAME}" \
  --ces_vector "${VEC}" \
  --mode boost --control all \
  --split test \
  --k_low "${K}" --k_full "${K}" \
  --max_samples "${N_TEST}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.0 --seed "${SEED}" \
  --ces_layer "${CES_LAYER}" --ces_coef "${COEF}" \
  --out_dir "${TEST_DIR}"

echo "=== done. reports: ${DEV_DIR}/report.json  ${TEST_DIR}/report.json ==="
