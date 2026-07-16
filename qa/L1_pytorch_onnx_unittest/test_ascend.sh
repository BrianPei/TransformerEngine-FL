#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L1_pytorch_onnx_unittest-ascend}"
: "${PYTHON_BIN:=python3}"
reason="The ONNX suite hardcodes CUDA tensors and CUDA custom operators, while ONNX Runtime in this environment has no Ascend execution provider."
"$PYTHON_BIN" "$TE_PATH/qa/ascend_write_junit_skip.py" --output "$XML_LOG_DIR/test_onnx_export.xml" --suite "L1_pytorch_onnx_unittest_ascend" --reason "$reason"
echo "[SKIP] $reason"
