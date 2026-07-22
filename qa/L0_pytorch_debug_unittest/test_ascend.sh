#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"

: "${XML_LOG_DIR:=$TE_PATH/logs/L0_pytorch_debug_unittest-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"

export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch
export TORCHDYNAMO_DISABLE=1
: "${NVTE_TEST_NVINSPECT_FEATURE_DIRS:=$TE_PATH/transformer_engine/debug/features}"
: "${NVTE_TEST_NVINSPECT_CONFIGS_DIR:=$TE_PATH/tests/pytorch/debug/test_configs/}"

if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install pytest==8.2.1
fi

FAIL=0

run_test_step() {
    local xml_file=$1
    local test_path=$2
    shift 2

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $test_path"
    "$PYTHON_BIN" "$TE_PATH/qa/ascend_run_pytest.py" \
        -v -s \
        --junitxml="$XML_LOG_DIR/$xml_file" \
        "$test_path" \
        "$@" || FAIL=1
}

run_test_step "test_sanity.xml" "$TE_PATH/tests/pytorch/debug/test_sanity.py" \
    --feature_dirs="$NVTE_TEST_NVINSPECT_FEATURE_DIRS"
run_test_step "test_config.xml" "$TE_PATH/tests/pytorch/debug/test_config.py" \
    --feature_dirs="$NVTE_TEST_NVINSPECT_FEATURE_DIRS"
run_test_step "test_numerics.xml" "$TE_PATH/tests/pytorch/debug/test_numerics.py" \
    --feature_dirs="$NVTE_TEST_NVINSPECT_FEATURE_DIRS"
run_test_step "test_log.xml" "$TE_PATH/tests/pytorch/debug/test_log.py" \
    --feature_dirs="$NVTE_TEST_NVINSPECT_FEATURE_DIRS" \
    --configs_dir="$NVTE_TEST_NVINSPECT_CONFIGS_DIR"
NVTE_TORCH_COMPILE=0 run_test_step "test_api_features.xml" \
    "$TE_PATH/tests/pytorch/debug/test_api_features.py" \
    --no-header \
    --feature_dirs="$NVTE_TEST_NVINSPECT_FEATURE_DIRS" \
    --configs_dir="$NVTE_TEST_NVINSPECT_CONFIGS_DIR"
run_test_step "test_perf.xml" "$TE_PATH/tests/pytorch/debug/test_perf.py" \
    --feature_dirs="$NVTE_TEST_NVINSPECT_FEATURE_DIRS" \
    --configs_dir="$NVTE_TEST_NVINSPECT_CONFIGS_DIR"

"$PYTHON_BIN" "$TE_PATH/qa/ascend_validate_junit.py" \
    --min-tests 6 \
    --min-passed 6 \
    "$XML_LOG_DIR/test_sanity.xml" \
    "$XML_LOG_DIR/test_config.xml" \
    "$XML_LOG_DIR/test_numerics.xml" \
    "$XML_LOG_DIR/test_log.xml" \
    "$XML_LOG_DIR/test_api_features.xml" \
    "$XML_LOG_DIR/test_perf.xml"

exit "$FAIL"
