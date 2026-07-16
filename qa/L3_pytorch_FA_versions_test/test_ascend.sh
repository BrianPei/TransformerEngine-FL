#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L3_pytorch_FA_versions_test-ascend}"
: "${PYTHON_BIN:=python3}"
reason="flash-attn 2.x/3.x and its version matrix compile CUDA kernels for NVIDIA SM architectures; they cannot be built or executed on Ascend."
"$PYTHON_BIN" "$TE_PATH/qa/ascend_write_junit_skip.py" --output "$XML_LOG_DIR/pytest.xml" --suite "L3_pytorch_FA_versions_test_ascend" --reason "$reason"
echo "[SKIP] $reason"
