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

if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install pytest==8.2.1
fi

feature_dirs="$TE_PATH/transformer_engine/debug/features"

write_skip() {
    local xml_file=$1
    local suite=$2
    local reason=$3

    "$PYTHON_BIN" "$TE_PATH/qa/ascend_write_junit_skip.py" \
        --output "$XML_LOG_DIR/$xml_file" \
        --suite "$suite" \
        --reason "$reason"
    echo "[SKIP] $suite: $reason"
}

"$PYTHON_BIN" -m pytest \
    -q \
    -p no:warnings \
    --junitxml="$XML_LOG_DIR/pytest_debug_config_ascend.xml" \
    "$TE_PATH/tests/pytorch/debug/test_config.py" \
    --feature_dirs="$feature_dirs"

reason="The remaining debug suite exercises CUDA-only FP8 kernels or CUDA numerics paths that are not supported by the Ascend backend."
write_skip "test_sanity.xml" "L0_pytorch_debug_unittest_ascend.test_sanity" "$reason"
write_skip "test_numerics.xml" "L0_pytorch_debug_unittest_ascend.test_numerics" "$reason"
write_skip "test_log.xml" "L0_pytorch_debug_unittest_ascend.test_log" "$reason"
write_skip "test_api_features.xml" "L0_pytorch_debug_unittest_ascend.test_api_features" "$reason"
write_skip "test_perf.xml" "L0_pytorch_debug_unittest_ascend.test_perf" "$reason"
write_skip "test_sanity_2.xml" "L0_pytorch_debug_unittest_ascend.test_sanity_2" "$reason"
write_skip "test_numerics_2.xml" "L0_pytorch_debug_unittest_ascend.test_numerics_2" "$reason"

echo "[INFO] Ascend debug coverage keeps the config path active and records unsupported CUDA debug paths as skipped."
