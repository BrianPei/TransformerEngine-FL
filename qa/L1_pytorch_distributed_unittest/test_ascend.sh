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

FAIL=0

"$PYTHON_BIN" "$TE_PATH/qa/ascend_run_pytest.py" -v -s \
    --junitxml="$XML_LOG_DIR/pytest_hccl_te.xml" \
    "$TE_PATH/qa/test_ascend_hccl_distributed.py" || FAIL=1

# MetaX runs this platform-independent Context Parallel utility suite as well.
"$PYTHON_BIN" -m pytest -v -s \
    --junitxml="$XML_LOG_DIR/pytest_test_cp_utils.xml" \
    "$TE_PATH/tests/pytorch/attention/test_cp_utils.py" || FAIL=1

NVTE_FLASH_ATTN=0 \
NVTE_FUSED_ATTN=0 \
NVTE_UNFUSED_ATTN=1 \
NVTE_ASCEND_DISTRIBUTED_NUMERICS_SUBSET=1 \
"$PYTHON_BIN" "$TE_PATH/qa/ascend_run_pytest.py" -v -s \
    --tb=short \
    --junitxml="$XML_LOG_DIR/pytest_distributed_numerics_none.xml" \
    "$TE_PATH/tests/pytorch/distributed/test_numerics.py::test_distributed[None]" || FAIL=1

exit "$FAIL"
