#!/usr/bin/env bash
# GPU-box bootstrap: install pins, assert a GPU, build the data splits.
set -euo pipefail
pip install --quiet "trl==1.12.0" "transformers==5.16.1" "datasets==5.0.1" accelerate pandas matplotlib
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no GPU visible — wrong box/image"
print("GPU:", torch.cuda.get_device_name(0))
PY
python scripts/prepare_data.py
echo "READY. Train with: python scripts/train_rm.py --seed 0 --frac 1.0"
