#!/usr/bin/env bash
# Tighten the Pareto ladder for paper-grade claims:
#   - n=100
#   - --sample_items so seeds draw different subsets (real multi-seed)
#   - seeds 42 and 123
#   - GSM8K gets a finer B grid; MATH/MedQA keep {16,32,64,128}
#   - bootstrap CIs written by exp_pareto_ladder.py
#
# Launch on pod:
#   tmux new -d -s pareto_t 'bash scripts/run_pareto_tighten.sh > /workspace/pareto_tighten.log 2>&1'
set -uo pipefail
cd /workspace/latentmas-baseline || exit 2

if [[ -f env_native.sh ]]; then
  # shellcheck disable=SC1091
  source env_native.sh
elif [[ -f /root/venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /root/venv/bin/activate
  export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
else
  echo "[pareto] no venv — run lean_recover_env.sh first" >&2
  exit 2
fi

OUT=${OUT:-artifacts/exp2x2/pareto_tight}
SEEDS=${SEEDS:-"42 123"}
N=${N:-100}
K=${K:-40}
BATCH=${BATCH:-16}
MODEL=${MODEL:-Qwen/Qwen3-14B}
WITH_NONE=${WITH_NONE:-1}
mkdir -p "$OUT"
SENTINEL="$OUT/STOP_POD"
rm -f "$SENTINEL"

budget_for_task () {
  case "$1" in
    math)  echo 2048 ;;
    medqa) echo 1024 ;;
    *)     echo 768  ;;
  esac
}
batch_for_task () {
  case "$1" in
    math|medqa) echo 12 ;;
    *)          echo "${BATCH}" ;;
  esac
}
budgets_for_task () {
  case "$1" in
    gsm8k) echo "16,24,32,48,64,128" ;;  # finer soft→free transition
    *)     echo "16,32,64,128" ;;
  esac
}

echo "=== pareto_tighten started $(date) ===" | tee -a "$OUT/chain.log"
nvidia-smi -L | tee -a "$OUT/chain.log"
python -c "import torch,transformers; print('torch', torch.__version__, 'tf', transformers.__version__, 'cuda', torch.cuda.is_available())" \
  | tee -a "$OUT/chain.log"

for seed in $SEEDS; do
  for task in gsm8k math medqa; do
    BUD=$(budget_for_task "$task")
    BS=$(batch_for_task "$task")
    BGRID=$(budgets_for_task "$task")
    tag="${task}_n${N}_s${seed}"
    echo "[pareto] START $tag budgets=$BGRID judger=$BUD batch=$BS $(date)" | tee -a "$OUT/chain.log"
    EXTRA=()
    if [[ "$WITH_NONE" == "1" ]]; then
      EXTRA+=(--with_none)
    fi
    python scripts/exp_pareto_ladder.py \
      --model_name "$MODEL" --task "$task" --split test \
      --k "$K" --n "$N" --batch_size "$BS" --judger_budget "$BUD" \
      --budgets "$BGRID" --sink 4 --importance key_norm \
      --sample_items --seed "$seed" \
      "${EXTRA[@]}" \
      --out_dir "$OUT/$tag" \
      && echo "[pareto] DONE $tag $(date)" | tee -a "$OUT/chain.log" \
      || echo "[pareto] FAIL $tag $(date)" | tee -a "$OUT/chain.log"
  done
done

echo "=== PARETO_TIGHTEN_DONE $(date) — STOP THE POD ===" | tee -a "$OUT/chain.log"
date > "$SENTINEL"
