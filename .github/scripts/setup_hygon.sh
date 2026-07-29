#!/usr/bin/env bash
# Hygon/DTK environment setup for TransformerEngine-FL plugin QA.
set -euo pipefail

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"

echo "===== Load Hygon/DTK runtime environment ====="
source "$WORKSPACE/qa/plugin/hygon/set_env.sh"

echo "===== Verify Hygon device visibility ====="
if command -v hy-smi >/dev/null 2>&1; then
    hy-smi || true
else
    echo "WARNING: hy-smi is unavailable in this environment"
fi

echo "===== Verify Python runtime ====="
"$PYTHON_BIN" - <<'PY'
import sys

print("python:", sys.executable)
print("version:", sys.version)

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(f"PyTorch is required in the Hygon CI image: {exc}") from exc

print("torch:", torch.__version__)
PY

echo "===== Install Hygon QA dependencies ====="
if [ "${HYGON_SKIP_DEP_INSTALL:-0}" = "1" ]; then
    echo "Skipping Python dependency installation because HYGON_SKIP_DEP_INSTALL=1"
else
    "$PYTHON_BIN" -m pip install pytest==8.2.1 expecttest

    if [ "${HYGON_INSTALL_ONNX_DEPS:-1}" = "1" ]; then
        "$PYTHON_BIN" -m pip install onnxruntime onnxruntime_extensions
    fi
fi

if [ "${HYGON_INSTALL_TE:-0}" = "1" ]; then
    echo "===== Install TransformerEngine-FL Python layer ====="
    cd "$WORKSPACE"
    TE_FL_SKIP_CUDA=1 "$PYTHON_BIN" setup.py install
else
    echo "Skipping TransformerEngine-FL install; tests run from source via PYTHONPATH"
fi

if [ -n "${GITHUB_ENV:-}" ]; then
    {
        echo "PATH=$PATH"
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
        echo "PYTHONPATH=$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
        echo "TE_PATH=$WORKSPACE"
        echo "XML_LOG_DIR=$WORKSPACE/logs"
        echo "PLATFORM=$PLATFORM"
        echo "TE_FL_SKIP_CUDA=$TE_FL_SKIP_CUDA"
        echo "TE_FL_PREFER=$TE_FL_PREFER"
        echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
        echo "PYTHON_BIN=$PYTHON_BIN"
        echo "NVTE_FLASH_ATTN=$NVTE_FLASH_ATTN"
        echo "NVTE_FUSED_ATTN=$NVTE_FUSED_ATTN"
        echo "NVTE_UNFUSED_ATTN=$NVTE_UNFUSED_ATTN"
        echo "NVTE_UnfusedDPA_Emulate_FP8=$NVTE_UnfusedDPA_Emulate_FP8"
    } >> "$GITHUB_ENV"
fi

echo "===== Hygon environment setup complete ====="
