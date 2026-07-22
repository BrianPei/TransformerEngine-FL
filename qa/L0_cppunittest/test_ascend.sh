#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L0_cppunittest-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"

export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch
TE_LIB_PATH=$("$PYTHON_BIN" - <<'PY'
import site

print(site.getsitepackages()[0])
PY
)
export LD_LIBRARY_PATH="$TE_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON_BIN" - <<'PY'
import torch

try:
    import torch_npu  # noqa: F401
except ImportError as exc:
    raise SystemExit(f"torch_npu is required for the Ascend test path: {exc}")

if not torch.npu.is_available():
    raise SystemExit("Ascend NPU is not available to PyTorch.")

print(f"[INFO] Ascend NPU devices visible to PyTorch: {torch.npu.device_count()}")
PY

if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install pytest==8.2.1
fi

failed=0

skip_test_point() {
    local label=$1
    local reason=$2

    echo "-------------------------------------------------------"
    echo "[SKIP] $label"
    echo "       $reason"
}

run_pytest_file() {
    local test_file=$1
    local test_name
    local pytest_exit

    if [ ! -f "$test_file" ]; then
        echo "Required Ascend test file not found: $test_file" >&2
        failed=1
        return
    fi

    test_name=$(basename "$test_file" .py)
    echo "-------------------------------------------------------"
    echo "[RUN] $test_name"

    set +e
    "$PYTHON_BIN" - "$test_file" "$XML_LOG_DIR/pytest_ascend_${test_name}.xml" <<'PY'
import sys

import pytest

test_file = sys.argv[1]
junit_xml = sys.argv[2]

try:
    import torch
    import torch_npu  # noqa: F401
    import transformer_engine

    transformer_engine.TE_DEVICE_TYPE = "npu"
    transformer_engine.TE_PLATFORM = torch_npu.npu
    torch.cuda.is_current_stream_capturing = lambda: False
except Exception as exc:
    print(f"[WARN] Failed to set NPU device type before pytest: {exc}", file=sys.stderr)

raise SystemExit(
    pytest.main(
        [
            "-q",
            "-x",
            "-p",
            "no:warnings",
            f"--junitxml={junit_xml}",
            test_file,
        ]
    )
)
PY
    pytest_exit=$?
    set -e

    if [ "$pytest_exit" -eq 5 ]; then
        echo "[SKIP] $test_name collected no pytest tests."
    elif [ "$pytest_exit" -ne 0 ]; then
        echo "[FAIL] $test_name exited with status $pytest_exit." >&2
        failed=1
    fi
}

skip_test_point \
    "$TE_PATH/tests/cpp ctest" \
    "tests/cpp is CUDA C++/TE ABI based; the Ascend path validates the corresponding plugin backend instead."

ascend_test_files=(
    "$TE_PATH/transformer_engine/plugin/tests/test_backend_ascend_smoke.py"
    "$TE_PATH/transformer_engine/plugin/tests/test_backend_flagos.py"
    "$TE_PATH/transformer_engine/plugin/tests/test_backend_reference.py"
    "$TE_PATH/transformer_engine/plugin/tests/test_backend_reference_activation.py"
    "$TE_PATH/transformer_engine/plugin/tests/test_backend_reference_dropout.py"
    "$TE_PATH/transformer_engine/plugin/tests/test_backend_reference_gemm.py"
    "$TE_PATH/transformer_engine/plugin/tests/test_plugin_manager.py"
    "$TE_PATH/transformer_engine/plugin/tests/test_plugin_policy.py"
    "$TE_PATH/transformer_engine/plugin/tests/test_policy.py"
)

skipped_plugin_tests=(
    "transformer_engine/plugin/tests/test_activations.py:no pytest cases collected in this file"
    "transformer_engine/plugin/tests/test_backend_flagos_fused_adam.py:test inputs are CPU tensors but FlagGems launches Ascend Triton kernels"
    "transformer_engine/plugin/tests/test_backend_flagos_gemm.py:test matrix includes CPU tensor paths that are invalid for Ascend Triton kernels"
    "transformer_engine/plugin/tests/test_backend_flagos_multi_tensor.py:test inputs are CPU tensors but FlagGems launches Ascend Triton kernels"
    "transformer_engine/plugin/tests/test_backend_flagos_rmsnorm.py:test inputs are CPU tensors but FlagGems launches Ascend Triton kernels"
    "transformer_engine/plugin/tests/test_backend_flagos_softmax.py:test inputs are CPU tensors but FlagGems launches Ascend Triton kernels"
    "transformer_engine/plugin/tests/test_flash_attention.py:no pytest cases collected in this file"
    "transformer_engine/plugin/tests/test_fused_rope.py:no pytest cases collected in this file"
    "transformer_engine/plugin/tests/test_normalization.py:no pytest cases collected in this file"
    "transformer_engine/plugin/tests/test_operations.py:no pytest cases collected in this file"
    "transformer_engine/plugin/tests/test_optimizer.py:no pytest cases collected in this file"
    "transformer_engine/plugin/tests/test_softmax.py:no pytest cases collected in this file"
    "transformer_engine/plugin/tests/test_te_general_grouped.py:no pytest cases collected in this file"
)

for test_file in "${ascend_test_files[@]}"; do
    run_pytest_file "$test_file"
done

for entry in "${skipped_plugin_tests[@]}"; do
    skip_test_point "${entry%%:*}" "${entry#*:}"
done

if [ "$failed" -ne 0 ]; then
    echo "One or more Ascend plugin test files failed." >&2
    exit 1
fi

echo "All Ascend plugin and NPU tests passed."
