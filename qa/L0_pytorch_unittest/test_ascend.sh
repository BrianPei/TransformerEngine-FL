#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L0_pytorch_unittest-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"

export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch

FAIL=0

run_test_step() {
    local xml_file=$1
    local test_path=$2
    local runner=${3:-pytest}
    local label=${4:-$(basename "$test_path")}
    local command=("$PYTHON_BIN" -m pytest)

    if [ "$runner" = "ascend" ]; then
        command=("$PYTHON_BIN" "$TE_PATH/qa/ascend_run_pytest.py")
    elif [ "$runner" = "isolated" ]; then
        # These backend files install flag_gems mocks during module import.
        # pytest-cov imports TE first and defeats that isolation on Ascend.
        command=(env -u PYTEST_ADDOPTS "$PYTHON_BIN" -m pytest)
    fi

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    "${command[@]}" \
        -v -s --tb=short \
        --junitxml="$XML_LOG_DIR/$xml_file" \
        "$test_path" || FAIL=1
}

echo "[INFO] Running real-NPU TE unit tests on Ascend."
run_test_step \
    "pytest_ascend_backend_smoke.xml" \
    "$TE_PATH/qa/test_backend_ascend_smoke.py" \
    ascend \
    "Ascend TE module smoke tests"
run_test_step \
    "pytest_ascend_backend_ops.xml" \
    "$TE_PATH/qa/test_backend_ascend_ops.py" \
    ascend \
    "Ascend FlagOS operator tests"

echo "[INFO] Running the Plugin Unit files shared with CUDA and MetaX."
PLUGIN_TEST_ROOT="$TE_PATH/transformer_engine/plugin/tests"
run_test_step "pytest_test_plugin_policy.xml" \
    "$PLUGIN_TEST_ROOT/test_plugin_policy.py"
run_test_step "pytest_test_plugin_manager.xml" \
    "$PLUGIN_TEST_ROOT/test_plugin_manager.py"
run_test_step "pytest_test_backend_flagos.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_flagos.py" isolated
run_test_step "pytest_test_backend_flagos_fused_adam.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_flagos_fused_adam.py" isolated
run_test_step "pytest_test_backend_flagos_gemm.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_flagos_gemm.py" isolated
run_test_step "pytest_test_backend_flagos_multi_tensor.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_flagos_multi_tensor.py" isolated
run_test_step "pytest_test_backend_flagos_rmsnorm.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_flagos_rmsnorm.py" isolated
run_test_step "pytest_test_backend_flagos_softmax.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_flagos_softmax.py" isolated
run_test_step "pytest_test_backend_reference.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_reference.py" isolated
run_test_step "pytest_test_backend_reference_activation.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_reference_activation.py" isolated
run_test_step "pytest_test_backend_reference_dropout.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_reference_dropout.py" isolated
run_test_step "pytest_test_backend_reference_gemm.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_reference_gemm.py" isolated

if [ "$FAIL" -ne 0 ]; then
    echo "Some Ascend PyTorch Unit tests failed."
    exit 1
fi

echo "All assigned Ascend PyTorch Unit tests passed (some might have been skipped)."
