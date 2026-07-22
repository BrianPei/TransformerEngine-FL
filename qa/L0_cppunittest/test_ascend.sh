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

# tests/cpp is implemented with CUDA source files and the CUDA TE ABI. The
# Ascend backend lives in the FlagOS plugin layer, so this entry point runs the
# applicable plugin tests plus a smoke test against the real NPU backend.
test_files=(
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

failed=0
for test_file in "${test_files[@]}"; do
    if [ ! -f "$test_file" ]; then
        echo "Required Ascend test file not found: $test_file" >&2
        failed=1
        continue
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
done

if [ "$failed" -ne 0 ]; then
    echo "One or more Ascend plugin test files failed." >&2
    exit 1
fi

echo "All Ascend plugin and NPU tests passed."
