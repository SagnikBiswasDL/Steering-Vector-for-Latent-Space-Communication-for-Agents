#!/usr/bin/env bash
# End-to-end native-vector pipeline (run on the RunPod pod, venv active).
#
# Stages:
#   1. Capture in-pipeline layer-L activations + correctness on the TRAIN split.
#   2. Build per-agent correctness-contrastive native vectors.
#   3. Error analysis (per-agent correctness probe + token/format taxonomy).
#   4. Evaluate {control, per-agent, upstream3, all} vs Judger-only on TEST.
#   5. Plot the accuracy-token Pareto frontier.
#
# Usage:
#   bash scripts/run_native_pipeline.sh gsm8k 400 300 "40 80"
#     args: TASK  CAPTURE_N  EVAL_N  "COEFS"
set -euo pipefail

TASK="${1:-gsm8k}"
CAP_N="${2:-400}"
EVAL_N="${3:-300}"
COEFS="${4:-40 80}"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
LAYER="${LAYER:-28}"
SEED="${SEED:-42}"
BS="${BS:-25}"
CAP_SPLIT="${CAP_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

ART=artifacts
CAP="${ART}/capture/${TASK}_${CAP_SPLIT}_s${SEED}.pt"
VDIR="${ART}/seal_vectors/qwen3-14b/native_${TASK}"
CSV="${ART}/sweeps/native_${TASK}.csv"
GENERIC="${GENERIC:-${ART}/seal_vectors/qwen3-14b/gsm8k_layer28.pt}"

echo "=== [1/5] Capture (${TASK}/${CAP_SPLIT}, n=${CAP_N}) ==="
python run.py --method latent_mas --model_name "${MODEL}" --task "${TASK}" \
  --prompt sequential --latent_steps 40 --max_new_tokens 2048 \
  --split "${CAP_SPLIT}" --max_samples "${CAP_N}" --generate_bs "${BS}" \
  --temperature 0.6 --top_p 0.95 --seed "${SEED}" \
  --capture_acts "${CAP}" --capture_layer "${LAYER}" > "logs_capture_${TASK}.txt" 2>&1 || \
  { echo "capture failed; see logs_capture_${TASK}.txt"; tail -n 30 "logs_capture_${TASK}.txt"; exit 1; }

echo "=== [2/5] Build native vectors ==="
python scripts/build_native_vectors.py --cache "${CAP}" --out_dir "${VDIR}"

echo "=== [3/5] Error analysis ==="
python scripts/analyze_pipeline_errors.py --cache "${CAP}" \
  --out_json "${ART}/analysis/errors_${TASK}.json" \
  --out_plot "${ART}/plots/probe_auc_${TASK}.png"

echo "=== [4/5] Native eval sweep (${TASK}/${EVAL_SPLIT}, n=${EVAL_N}) ==="
GEN_FLAG=""
if [ -f "${GENERIC}" ]; then GEN_FLAG="--judger_generic ${GENERIC}"; fi
python scripts/native_eval_sweep.py --model_name "${MODEL}" --task "${TASK}" \
  --split "${EVAL_SPLIT}" --n "${EVAL_N}" --native_dir "${VDIR}" \
  --coefs ${COEFS} \
  --groups control planner critic refiner judger upstream3 all \
  --latent_steps 40 --max_new_tokens 2048 --generate_bs "${BS}" \
  --seed "${SEED}" --out_csv "${CSV}" ${GEN_FLAG}

echo "=== [5/5] Pareto plot ==="
python scripts/plot_pareto.py --csv "${CSV}" \
  --out_plot "${ART}/plots/pareto_${TASK}.png" --title "${TASK} (n=${EVAL_N})"

echo "=== DONE. Artifacts in ${ART}/ ==="
