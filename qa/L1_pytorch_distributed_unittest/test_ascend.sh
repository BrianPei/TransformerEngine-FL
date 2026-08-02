#!/usr/bin/env bash

set -u

: "${TE_PATH:=${GITHUB_WORKSPACE:-$(pwd)}}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L1_pytorch_distributed_unittest-ascend}"
mkdir -p "$XML_LOG_DIR"

FAIL=0

test_fail() {
    FAIL=1
    echo "Error: sub-test failed: $1"
}

run_pytest_step() {
    local label=$1
    local junit=$2
    shift 2

    local cmd=(python3 -m pytest)
    cmd+=(-v -s --tb=short "--junitxml=$XML_LOG_DIR/$junit")
    cmd+=("$@")

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    "${cmd[@]}" || test_fail "$label"
}

if python3 - <<'PY'
import importlib.util

required = ("torch", "transformer_engine")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Skipping context parallel utilities; missing modules: " + ", ".join(missing))
    raise SystemExit(1)
PY
then
    run_pytest_step "context parallel utilities" "pytest_test_cp_utils.xml" \
        "$TE_PATH/tests/pytorch/attention/test_cp_utils.py"
fi

NVTE_FLASH_ATTN=0 \
NVTE_FUSED_ATTN=0 \
NVTE_UNFUSED_ATTN=1 \
    run_pytest_step "distributed non-FP8 numerics" "pytest_distributed_numerics_none.xml" \
        "$TE_PATH/tests/pytorch/distributed/test_numerics.py::test_ascend_distributed_smoke"

echo "Skipping Ascend HCCL communication tests."

if [ "$FAIL" -ne 0 ]; then
    echo "Some tests failed."
    exit 1
fi
