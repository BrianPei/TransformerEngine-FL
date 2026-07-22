#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L1_pytorch_thunder_integration-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"

reason="Lightning Thunder's Transformer Engine executor is CUDA/FP8-specific and /opt/pytorch/lightning-thunder is not installed."
"$PYTHON_BIN" "$TE_PATH/qa/ascend_write_junit_skip.py" \
    --output "$XML_LOG_DIR/pytest.xml" \
    --suite "L1_pytorch_thunder_integration_ascend" \
    --reason "$reason"
echo "[SKIP] $reason"
