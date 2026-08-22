#!/bin/bash
# Paper matrix: the hardening + breadth runs for the "latent relay = content-independent
# scaffold" thesis. Per (task, seed) runs one batched job with arms:
#   A full/noASC, B full/ASC, C obf/noASC, D obf/ASC, E evict/noASC, F evict/ASC, S shuffled/noASC
# which covers, in a single harness:
#   - mechanism crux:   A vs S (real vs wrong-question cache)  [content-independence]
#   - compression:      A vs C/E (free? verbosity tax?)
#   - ASC complement:   B-A, D-C
# Runs detached-friendly (foreground here; wrap in tmux/nohup as needed).
#
# Usage (on pod):  bash scripts/run_paper_matrix.sh
# Env knobs: TASKS, SEEDS, N, VEC, ASC_COEF, RELAY_BUDGET, BATCH
set -u
cd /workspace/latentmas-baseline || exit 2
source /root/venv/bin/activate
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}

MODEL=${MODEL:-Qwen/Qwen3-14B}
TASKS=${TASKS:-"gsm8k medqa math"}
SEEDS=${SEEDS:-"42 123 7"}
N=${N:-100}
BATCH=${BATCH:-20}
RELAY_BUDGET=${RELAY_BUDGET:-16}
SINK=${SINK:-4}; RANK=${RANK:-8}
VEC=${VEC:-artifacts/seal_vectors/qwen3-14b/gsm8k_layer28_n200.pt}
ASC_COEF=${ASC_COEF:-40}
OUT=${OUT:-artifacts/exp2x2/paper}

budget_for () {  # per-task Judger budget (MATH needs long CoT)
  case "$1" in
    math)  echo 2048 ;;
    medqa) echo 1024 ;;
    *)     echo 768  ;;
  esac
}

for task in $TASKS; do
  BUD=$(budget_for "$task")
  for seed in $SEEDS; do
    tag="${task}_b${RELAY_BUDGET}_s${seed}"
    echo "=== [paper] $tag (budget=$BUD) ==="
    python scripts/exp_2x2_relay_asc.py \
      --model_name "$MODEL" --task "$task" --split test \
      --k 40 --n "$N" --batch_size "$BATCH" --judger_budget "$BUD" \
      --relay_budget "$RELAY_BUDGET" --sink "$SINK" --rank "$RANK" --importance key_norm \
      --with_evict --with_shuffled \
      --asc_vector "$VEC" --asc_coef "$ASC_COEF" --seed "$seed" \
      --out_dir "$OUT/$tag" \
      && echo "DONE $tag" || echo "FAIL $tag"
  done
done
echo "PAPER_MATRIX_DONE"
