#!/usr/bin/env bash
# Huawei Ascend NPU environment setup for TransformerEngine-FL.
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

export PLATFORM="${PLATFORM:-ascend}"
export TE_FL_SKIP_CUDA="${TE_FL_SKIP_CUDA:-1}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export NVTE_WITH_CUDA="${NVTE_WITH_CUDA:-0}"
export NVTE_WITH_MACA="${NVTE_WITH_MACA:-0}"
export TE_WITH_NCCL="${TE_WITH_NCCL:-0}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1,2,3}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

echo "===== Activate Python environment ====="
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-base}"
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV:-flagscale-train}"
else
    echo "WARNING: No supported conda installation found; using current environment"
fi

echo "===== Load Ascend runtime environment ====="
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/latest/set_env.sh ]; then
    source /usr/local/Ascend/latest/set_env.sh
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    # Persist the active runtime and pytest bootstrap for subsequent CI steps.
    {
        echo "PLATFORM=$PLATFORM"
        echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
        echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
        echo "NVTE_WITH_CUDA=$NVTE_WITH_CUDA"
        echo "NVTE_WITH_MACA=$NVTE_WITH_MACA"
        echo "TE_WITH_NCCL=$TE_WITH_NCCL"
        echo "ASCEND_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES"
        echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
        echo "PYTORCH_NPU_ALLOC_CONF=$PYTORCH_NPU_ALLOC_CONF"
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        echo "TE_TEST_PYTEST_COMMAND=python3 $WORKSPACE/tests/plugin/backend/npu/run_pytest.py"
    } >> "$GITHUB_ENV"
fi

echo "===== Verify Ascend PyTorch runtime ====="
python3 - <<'PY'
import torch
import torch_npu  # noqa: F401

print("torch:", torch.__version__)

if not hasattr(torch, "npu"):
    raise SystemExit("PyTorch NPU API is unavailable")

if not torch.npu.is_available():
    raise SystemExit("Ascend NPU is not available")

print("NPU device count:", torch.npu.device_count())
PY

echo "===== Verify test dependencies ====="
python3 - <<'PY'
try:
    import nvdlfw_inspect  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(f"nvdlfw_inspect is required for Ascend tests: {exc}") from exc
PY

echo "===== Install TransformerEngine-FL Python/plugin layer ====="
cd "$WORKSPACE"
python3 -m pip uninstall -y transformer_engine transformer_engine_torch || true
TE_FL_SKIP_CUDA=1 python3 setup.py install

echo "===== Verify TransformerEngine installation ====="
python3 tests/pytorch/test_sanity_import.py

echo "===== Ascend environment setup complete ====="
