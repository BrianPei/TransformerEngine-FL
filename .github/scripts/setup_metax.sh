#!/usr/bin/env bash
# Metax Platform Environment Setup Script
# Called by unit_tests_common.yml for Metax platforms (C500, etc.)
set -euo pipefail

echo "===== Step 0: Activate Python environment ====="
source /opt/conda/etc/profile.d/conda.sh
conda activate base
echo "PATH=$PATH" >> $GITHUB_ENV
echo "Python: $(which python3) ($(python3 --version 2>&1))"

echo "===== Step 1: Base Environment Setup ====="
# Configure MACA toolchain paths
export PATH=/opt/maca/bin:$PATH
export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
service ssh restart

echo "===== Step 2: Create nvcc Symlink (cucc -> nvcc) ====="
# TransformerEngine expects nvcc, but MACA provides cucc
ln -sf /opt/maca/tools/cu-bridge/bin/cucc /opt/maca/tools/cu-bridge/bin/nvcc
which nvcc || true

echo "===== Step 3: Install Required System Tools ====="
# Use apt to install git, curl
sed -i 's|http://mirrors.aliyun.com/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list
apt-get update -qq || true
apt-get install -y -qq git curl
# Install cmake and ninja via pip (more reliable than apt in this env)
python3 -m pip install cmake ninja torch --no-cache-dir

echo "===== Step 4: Remove Existing TransformerEngine ====="
# Prevent conflicts with preinstalled or incompatible versions
python3 -m pip uninstall transformer_engine -y || true
python3 -m pip install nvdlfw-inspect --no-deps || true

echo "===== Step 5: Install TE-FL Plugin Layer ====="
# Install TransformerEngine-FL Python layer (plugin logic)
cd $GITHUB_WORKSPACE
TE_FL_SKIP_CUDA=1 python3 setup.py install

echo "===== Step 6: Final Verification ====="
# Verify both TE Python API and backend are functional
python3 - <<'EOF'
import transformer_engine
import transformer_engine_torch as te
print("transformer_engine:", transformer_engine)
print("transformer_engine_torch:", te)
EOF

echo "===== Environment Setup Complete ====="
