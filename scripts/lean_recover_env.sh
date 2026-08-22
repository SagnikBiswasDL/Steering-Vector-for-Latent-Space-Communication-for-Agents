#!/usr/bin/env bash
# Fast pod env rebuild after /root wipe. Skips vllm (HF-only harness).
set -euo pipefail
mkdir -p /root/tmp /root/.cache/pip
export TMPDIR=/root/tmp PIP_CACHE_DIR=/root/.cache/pip
cd /workspace/latentmas-baseline

echo "[lean] creating /root/venv (system-site-packages for torch)"
python3 -m venv --system-site-packages /root/venv
# shellcheck disable=SC1091
source /root/venv/bin/activate
pip install --upgrade pip
echo "[lean] installing HF stack (pinned transformers)"
pip install "transformers==4.57.1" datasets accelerate numpy tqdm matplotlib hf_transfer
python -c "import torch,transformers,datasets; print('[lean] OK torch', torch.__version__, 'tf', transformers.__version__, 'cuda', torch.cuda.is_available())"
echo LEAN_RECOVER_DONE
