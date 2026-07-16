#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L1_jax_distributed_unittest-ascend}"
: "${PYTHON_BIN:=python3}"
reason="L1 JAX distributed tests require CUDA PTXAS, XLA GPU and NCCL; no supported JAX Ascend runtime is installed."
"$PYTHON_BIN" "$TE_PATH/qa/ascend_write_junit_skip.py" --output "$XML_LOG_DIR/pytest.xml" --suite "L1_jax_distributed_unittest_ascend" --reason "$reason"
echo "[SKIP] $reason"
