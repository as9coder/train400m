#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" &>/dev/null; then
  echo "Python not found. Set PYTHON=/path/to/python3"
  exit 1
fi

"$PYTHON" -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip

# Install PyTorch with CUDA (cu124/cu126 wheels — match your driver; H100 works with recent cu12x)
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
  :
else
  echo "Installing PyTorch CUDA 12.4 wheels (edit URL for cu126 or another bundle if needed)."
  pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

pip install -r requirements.txt

echo ""
echo "H100 / max throughput: run training with --fast (uses BF16, larger batches, compile when possible)."
echo "Strongly recommended on Hopper: FlashAttention-2 (big attention speedup)"
echo "  pip install ninja packaging && pip install flash-attn --no-build-isolation"
echo ""
echo "Activate later: source .venv/bin/activate"
