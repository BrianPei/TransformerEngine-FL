#!/usr/bin/env bash
# MUSA Platform Environment Setup Script
# Called by unit_tests_common.yml / integration_tests_common.yml for MUSA platforms.
set -euo pipefail

echo "===== Step 0: Base Environment ====="
echo "Python: $(which python3) ($(python3 --version 2>&1))"
export PATH=/usr/local/musa/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}
export MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
export CUDA_HOME=${CUDA_HOME:-/usr/local/musa}
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export SKIP_CUDA_BUILD="${SKIP_CUDA_BUILD:-1}"
export NVTE_WITH_CUDA="${NVTE_WITH_CUDA:-0}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-0}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export TE_FL_ENABLE_MUSA_CUDA_COMPAT="${TE_FL_ENABLE_MUSA_CUDA_COMPAT:-1}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
        echo "MUSA_HOME=$MUSA_HOME"
        echo "CUDA_HOME=$CUDA_HOME"
        echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
        echo "SKIP_CUDA_BUILD=$SKIP_CUDA_BUILD"
        echo "NVTE_WITH_CUDA=$NVTE_WITH_CUDA"
        echo "NVTE_WITH_MACA=$NVTE_WITH_MACA"
        echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
        echo "TE_FL_ENABLE_MUSA_CUDA_COMPAT=$TE_FL_ENABLE_MUSA_CUDA_COMPAT"
        echo "TORCH_DEVICE_BACKEND_AUTOLOAD=$TORCH_DEVICE_BACKEND_AUTOLOAD"
    } >> "$GITHUB_ENV"
fi

echo "===== Step 1: Verify Image Dependencies ====="
python3 - <<'PY'
from importlib import metadata

required = (
    "pytest",
    "expecttest",
    "nvdlfw-inspect",
    "onnxruntime",
    "onnxruntime-extensions",
    "coverage",
    "pytest-cov",
)
missing = []
for package in required:
    try:
        print(f"{package}=={metadata.version(package)}")
    except metadata.PackageNotFoundError:
        missing.append(package)

if missing:
    raise RuntimeError(f"Missing MUSA CI image dependencies: {', '.join(missing)}")
PY

echo "===== Step 2: Verify Checked-out TransformerEngine-FL Python Layer ====="
cd "${GITHUB_WORKSPACE}"
python3 -c "import transformer_engine; print('transformer_engine:', transformer_engine.__file__)"

echo "===== Step 3: Verify MUSA Runtime ====="
python3 - <<'PY'
import importlib

import transformer_engine

tex = importlib.import_module("transformer_engine_musa_torch")
print("transformer_engine:", transformer_engine.__file__)
print("transformer_engine_musa_torch:", tex.__file__)
print("has multi_tensor_scale:", hasattr(tex, "multi_tensor_scale"))
print(
    "has multi_tensor_compute_scale_and_scale_inv:",
    hasattr(tex, "multi_tensor_compute_scale_and_scale_inv"),
)
PY

echo "===== MUSA Environment Setup Complete ====="
