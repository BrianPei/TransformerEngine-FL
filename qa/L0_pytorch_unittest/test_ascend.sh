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

result_xml="$XML_LOG_DIR/pytest_ascend_pytorch_unit.xml"

echo "[INFO] Running real-NPU TE and plugin unit tests on Ascend."
"$PYTHON_BIN" "$TE_PATH/qa/ascend_run_pytest.py" \
    -v -s --tb=short \
    --junitxml="$result_xml" \
    "$TE_PATH/qa/test_backend_ascend_smoke.py" \
    "$TE_PATH/qa/test_backend_ascend_ops.py" \
    "$TE_PATH/transformer_engine/plugin/tests/test_plugin_manager.py" \
    "$TE_PATH/transformer_engine/plugin/tests/test_plugin_policy.py" \
    "$TE_PATH/transformer_engine/plugin/tests/test_policy.py"

"$PYTHON_BIN" "$TE_PATH/qa/ascend_validate_junit.py" \
    --min-tests 10 \
    --min-passed 10 \
    --max-skipped 0 \
    "$result_xml"
