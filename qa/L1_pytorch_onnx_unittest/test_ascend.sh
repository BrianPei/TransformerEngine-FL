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

mkdir -p "$XML_LOG_DIR"

"$PYTHON_BIN" - <<'PY'
missing = []
for module in ("onnxruntime", "onnxruntime_extensions"):
    try:
        __import__(module)
    except ModuleNotFoundError:
        missing.append(module)
if missing:
    raise SystemExit(
        "Missing ONNX test dependencies in the CI image: " + ", ".join(missing)
    )
PY

NVTE_UnfusedDPA_Emulate_FP8=1 \
"$PYTHON_BIN" "$TE_PATH/qa/ascend_run_pytest.py" \
    -v -s \
    --tb=auto \
    --junitxml="$XML_LOG_DIR/test_onnx_export.xml" \
    "$TE_PATH/tests/pytorch/test_onnx_export.py"
