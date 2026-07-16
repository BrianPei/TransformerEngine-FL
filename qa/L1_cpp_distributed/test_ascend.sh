#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L1_cpp_distributed-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"
export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch

echo "[INFO] CUDA C++ comm_gemm has no Ascend build; running the real two-NPU HCCL/TE equivalent."
"$PYTHON_BIN" -m pytest -v -s \
    --junitxml="$XML_LOG_DIR/pytest_hccl_te.xml" \
    "$TE_PATH/qa/test_ascend_hccl_distributed.py"
