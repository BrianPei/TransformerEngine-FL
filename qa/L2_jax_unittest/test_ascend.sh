#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L2_jax_unittest-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"

reason="L2 JAX tests and examples require XLA GPU custom calls; no supported JAX Ascend runtime or TE JAX-NPU extension is installed."
"$PYTHON_BIN" "$TE_PATH/qa/ascend_write_junit_skip.py" \
    --output "$XML_LOG_DIR/pytest.xml" \
    --suite "L2_jax_unittest_ascend" \
    --reason "$reason"
echo "[SKIP] $reason"
