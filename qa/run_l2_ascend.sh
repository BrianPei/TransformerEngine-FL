#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
suites=(L2_jax_distributed_unittest L2_jax_unittest)
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
    else results+=("$suite: SKIPPED (no JAX Ascend runtime)"); fi
done
echo "======================================================================"
echo "Ascend L2 summary"
for result in "${results[@]}"; do echo "  $result"; done
exit "$failed"
