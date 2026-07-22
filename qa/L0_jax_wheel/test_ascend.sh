#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L0_jax_wheel-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"

reason="The project has no validated JAX Ascend runtime or native JAX-NPU extension, so a functional JAX Ascend wheel cannot be tested."
"$PYTHON_BIN" "$TE_PATH/qa/ascend_write_junit_skip.py" \
    --output "$XML_LOG_DIR/pytest_jax_wheel_ascend.xml" \
    --suite "L0_jax_wheel_ascend" \
    --reason "$reason"
echo "[SKIP] $reason"
