#!/usr/bin/env bash
# Memory–accuracy Pareto ladder across keep-budgets.
# One upstream build per item; evict@B for B in BUDGETS + full (+ optional none).
#
# Launch on pod:
#   tmux new -d -s pareto 'bash scripts/run_pareto_ladder.sh > /workspace/pareto_ladder.log 2>&1'
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
  echo "[pareto] no venv — run /workspace/recover_env.sh (or lean recover) first" >&2
  exit 2
fi

OUT=${OUT:-artifacts/exp2x2/pareto}
BUDGETS=${BUDGETS:-16,32,64,128}
TASKS=${TASKS:-"gsm8k math medqa"}
N=${N:-60}
SEED=${SEED:-42}
K=${K:-40}
BATCH=${BATCH:-16}
MODEL=${MODEL:-Qwen/Qwen3-14B}
WITH_NONE=${WITH_NONE:-1}
mkdir -p "$OUT"
SENTINEL="$OUT/STOP_POD"
rm -f "$SENTINEL"

budget_for () {
  case "$1" in
    math)  echo 2048 ;;
    medqa) echo 1024 ;;
    *)     echo 768  ;;
  esac
}
batch_for () {
  case "$1" in
    math)  echo 12 ;;
    medqa) echo 12 ;;
    *)     echo "${BATCH}" ;;
  esac
}

echo "=== pareto_ladder started $(date) ===" | tee -a "$OUT/chain.log"
nvidia-smi -L | tee -a "$OUT/chain.log"
python -c "import torch,transformers; print('torch', torch.__version__, 'tf', transformers.__version__, 'cuda', torch.cuda.is_available())" \
  | tee -a "$OUT/chain.log"

for task in $TASKS; do
  BUD=$(budget_for "$task")
  BS=$(batch_for "$task")
  tag="${task}_s${SEED}"
  echo "[pareto] START $tag budgets=$BUDGETS judger=$BUD batch=$BS $(date)" | tee -a "$OUT/chain.log"
  EXTRA=()
  if [[ "$WITH_NONE" == "1" ]]; then
    EXTRA+=(--with_none)
  fi
  python scripts/exp_pareto_ladder.py \
    --model_name "$MODEL" --task "$task" --split test \
    --k "$K" --n "$N" --batch_size "$BS" --judger_budget "$BUD" \
    --budgets "$BUDGETS" --sink 4 --importance key_norm \
    --seed "$SEED" \
    "${EXTRA[@]}" \
    --out_dir "$OUT/$tag" \
    && echo "[pareto] DONE $tag $(date)" | tee -a "$OUT/chain.log" \
    || echo "[pareto] FAIL $tag $(date)" | tee -a "$OUT/chain.log"
done

echo "=== PARETO_LADDER_DONE $(date) — STOP THE POD ===" | tee -a "$OUT/chain.log"
date > "$SENTINEL"
