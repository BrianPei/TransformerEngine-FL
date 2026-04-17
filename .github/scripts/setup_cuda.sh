#!/usr/bin/env bash
# CUDA Platform Environment Setup Script
# Called by unit_tests_common.yml for CUDA platforms (A100, H100, etc.)
set -euo pipefail

echo "===== Step 0: Activate Python environment ====="
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate flagscale-train
echo "PATH=$PATH" >> $GITHUB_ENV
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

echo "===== Environment Setup Complete ====="
