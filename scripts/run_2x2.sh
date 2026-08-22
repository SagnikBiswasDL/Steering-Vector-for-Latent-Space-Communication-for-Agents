#!/bin/bash
# Driver for the {Full, H-OBF} x {ASC off, ASC on} relay-compression experiment.
# Phase 1: 12-example smoke gate -> Phase 2: 100-example paired 2x2.
# Runs detached with a per-phase sentinel so it survives ssh disconnects.
#
# Usage (on the pod):
#   bash scripts/run_2x2.sh smoke     # 12-example gate only
#   bash scripts/run_2x2.sh paired    # 100-example paired run
#   bash scripts/run_2x2.sh all       # smoke, then paired iff smoke's gate passes
#
# Env: expects /root/venv + HF_HOME on /workspace.
set -u

REPO=/workspace/latentmas-baseline
VENV=/root/venv/bin/activate
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
cd "$REPO" || exit 2
source "$VENV"

MODEL=${MODEL:-Qwen/Qwen3-14B}
TASK=${TASK:-gsm8k}
K=${K:-40}
BUDGET=${BUDGET:-512}
RELAY_BUDGET=${RELAY_BUDGET:-32}
SINK=${SINK:-4}
RANK=${RANK:-8}
IMPORTANCE=${IMPORTANCE:-key_norm}
ASC_VECTOR=${ASC_VECTOR:-artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt}
ASC_COEF=${ASC_COEF:-40}
OUT=${OUT:-artifacts/exp2x2}
EXTRA=${EXTRA:-}   # e.g. EXTRA="--with_evict"

run_phase () {
  local tag=$1; shift
  local sentinel=/workspace/exp2x2_${tag}.sentinel
  local log=/workspace/exp2x2_${tag}.log
  rm -f "$sentinel" "$log"
  echo "[run_2x2] launching phase=$tag -> log=$log"
  nohup bash -c "python scripts/exp_2x2_relay_asc.py $* ; echo EXIT=\$? > $sentinel" > "$log" 2>&1 &
  echo "[run_2x2] pid=$! sentinel=$sentinel"
}

COMMON="--model_name $MODEL --task $TASK --k $K --judger_budget $BUDGET \
  --relay_budget $RELAY_BUDGET --sink $SINK --rank $RANK --importance $IMPORTANCE \
  --asc_vector $ASC_VECTOR --asc_coef $ASC_COEF $EXTRA"

case "${1:-all}" in
  smoke)
    run_phase smoke $COMMON --n 12 --smoke --out_dir $OUT/smoke ;;
  paired)
    run_phase paired $COMMON --n 100 --out_dir $OUT/paired ;;
  all)
    run_phase smoke $COMMON --n 12 --smoke --out_dir $OUT/smoke
    echo "[run_2x2] waiting on smoke gate before paired..."
    while [ ! -f /workspace/exp2x2_smoke.sentinel ]; do sleep 5; done
    if grep -q "proceed to 100-example" /workspace/exp2x2_smoke.log; then
      echo "[run_2x2] smoke gate PASSED -> launching paired"
      run_phase paired $COMMON --n 100 --out_dir $OUT/paired
    else
      echo "[run_2x2] smoke gate did NOT pass cleanly -- inspect /workspace/exp2x2_smoke.log"
    fi ;;
  *)
    echo "usage: $0 {smoke|paired|all}"; exit 1 ;;
esac
