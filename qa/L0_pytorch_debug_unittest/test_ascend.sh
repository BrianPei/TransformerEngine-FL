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

export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch

if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install pytest==8.2.1
fi

feature_dirs="$TE_PATH/transformer_engine/debug/features"
"$PYTHON_BIN" -m pytest \
    -q \
    -p no:warnings \
    --junitxml="$XML_LOG_DIR/pytest_debug_config_ascend.xml" \
    "$TE_PATH/tests/pytorch/debug/test_config.py" \
    --feature_dirs="$feature_dirs"

echo "[INFO] CUDA-only FP8 debug kernel tests are excluded from the Ascend configuration test."
