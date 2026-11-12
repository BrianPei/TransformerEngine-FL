#!/usr/bin/env bash
# Independent GPU baseline for the MUSA 5.2 dependency image.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="${XML_LOG_DIR:-$ROOT/logs/musa-portable}"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PLATFORM=mthreads TE_FL_SKIP_CUDA=1 SKIP_CUDA_BUILD=1
export NVTE_WITH_CUDA=0 NVTE_WITH_MACA=0 NVTE_WITH_NCCL_EP=0
export NVTE_FRAMEWORK=pytorch TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TE_FL_ENABLE_MUSA_CUDA_COMPAT=1
export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 NVTE_UNFUSED_ATTN=1
# torch_musa's Inductor benchmark imports triton.backends.mtgpu, absent from
# this FlagTree build. Keep TE's optional torch.compile fusion disabled.
# FlagOS kernels still compile and execute through FlagTree normally.
export NVTE_TORCH_COMPILE=0
# Clear inherited per-op overrides; each case asserts the actual implementation.
export TE_FL_PER_OP= TE_FL_DENY_VENDORS=musa

status=0
for backend in "$@"; do
    case "$backend" in
        reference|flagos) ;;
        *) echo "Unsupported backend: $backend" >&2; exit 2 ;;
    esac
done
if [ "$#" -eq 0 ]; then
    set -- reference flagos
fi
for backend in "$@"; do
    echo "Running real MUSA GPU checks: $backend"
    TE_FL_PREFER="$backend" "$PYTHON_BIN" -m pytest -s -v --tb=short \
        "$SCRIPT_DIR/test_gpu.py" \
        --junitxml="$LOG_DIR/$backend.xml" 2>&1 | tee "$LOG_DIR/$backend.log" || status=1
done
exit "$status"
