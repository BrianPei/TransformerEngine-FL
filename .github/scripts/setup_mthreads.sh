#!/usr/bin/env bash
# MooreThreads MUSA Platform Environment Setup Script
# Called by unit_tests_common.yml / integration_tests_common.yml for MUSA platforms.
set -euo pipefail

echo "===== Step 0: Base Environment ====="
echo "Python: $(which python3) ($(python3 --version 2>&1))"
export PATH=/usr/local/musa/bin:${PATH}
export LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/musa/lib:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}
export MUSA_HOME=${MUSA_HOME:-/usr/local/musa}
export CUDA_HOME=${CUDA_HOME:-/usr/local/musa}

echo "===== Step 1: Install Required Python Tools ====="
python3 -m pip install --no-cache-dir nvdlfw-inspect expecttest || true

echo "===== Step 2: Install TransformerEngine-FL Python Layer ====="
cd "${GITHUB_WORKSPACE}"
TE_FL_SKIP_CUDA=1 \
SKIP_CUDA_BUILD=1 \
NVTE_FRAMEWORK=pytorch \
python3 -m pip install --no-build-isolation --no-cache-dir -e .

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

echo "===== MThreads Environment Setup Complete ====="
