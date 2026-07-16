#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
all_suites=(L1_cpp_distributed L1_jax_distributed_unittest L1_pytorch_distributed_unittest L1_pytorch_mcore_integration L1_pytorch_onnx_unittest L1_pytorch_thunder_integration)
suites=("${all_suites[@]}")
if [ "$#" -gt 0 ]; then suites=("$@"); fi
failed=0
results=()
for suite in "${suites[@]}"; do
    echo "======================================================================"
    echo "[RUN] $suite"
    set +e
    TE_PATH="$TE_PATH" bash "$TE_PATH/qa/$suite/test_ascend.sh"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then results+=("$suite: FAILED (exit $rc)"); failed=1
    elif [[ "$suite" == L1_cpp_distributed || "$suite" == L1_pytorch_distributed_unittest ]]; then results+=("$suite: PASSED")
    else results+=("$suite: SKIPPED (unsupported dependency/backend)"); fi
done
echo "======================================================================"
echo "Ascend L1 summary"
for result in "${results[@]}"; do echo "  $result"; done
exit "$failed"
