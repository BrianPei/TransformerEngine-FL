#!/usr/bin/env bash
# CUDA Platform Environment Setup Script
# Called by unit_tests_common.yml for CUDA platforms (A100, H100, etc.)
set -euo pipefail

echo "===== Step 0: Activate Python environment ====="
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate flagscale-train
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib:${LD_LIBRARY_PATH:-}"
{
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
} >> "$GITHUB_ENV"
echo "Python: $(which python3) ($(python3 --version 2>&1))"

echo "===== Step 1: Remove Existing TransformerEngine ====="
pip uninstall transformer_engine transformer_engine_torch -y || true

echo "===== Step 2: Build & Install TransformerEngine ====="
cd $GITHUB_WORKSPACE

pip install nvdlfw-inspect --quiet
pip install expecttest --quiet
pip install . -v --no-deps --no-build-isolation

echo "===== Step 3: Verify Installation ====="
python3 tests/pytorch/test_sanity_import.py
python3 - <<'PY'
import importlib

tex = importlib.import_module("transformer_engine_torch")
required = ["multi_tensor_scale", "multi_tensor_compute_scale_and_scale_inv"]
missing = [name for name in required if not hasattr(tex, name)]
print("[TE check] module:", tex)
print("[TE check] file:", getattr(tex, "__file__", "N/A"))
print("[TE check] missing:", ", ".join(missing) if missing else "none")
if missing:
    raise SystemExit(1)
PY

echo "===== Environment Setup Complete ====="
