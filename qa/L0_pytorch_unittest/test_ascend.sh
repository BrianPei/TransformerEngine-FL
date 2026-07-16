#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L0_pytorch_unittest-ascend}"

export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch

echo "[INFO] Running the shared real-NPU and plugin test suite for PyTorch on Ascend."
TE_PATH="$TE_PATH" XML_LOG_DIR="$XML_LOG_DIR" \
    bash "$TE_PATH/qa/L0_cppunittest/test_ascend.sh"
