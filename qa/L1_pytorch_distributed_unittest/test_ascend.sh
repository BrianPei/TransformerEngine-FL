#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L1_pytorch_distributed_unittest-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"
export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch

result_xml="$XML_LOG_DIR/pytest_hccl_te.xml"
"$PYTHON_BIN" "$TE_PATH/qa/ascend_run_pytest.py" -v -s \
    --junitxml="$result_xml" \
    "$TE_PATH/qa/test_ascend_hccl_distributed.py"
