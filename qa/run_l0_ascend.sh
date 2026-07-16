#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"

all_suites=(
    L0_cppunittest
    L0_jax_distributed_unittest
    L0_jax_lint
    L0_jax_unittest
    L0_jax_wheel
    L0_license
    L0_pytorch_debug_unittest
    L0_pytorch_lint
    L0_pytorch_unittest
    L0_pytorch_wheel
)

if [ "$#" -gt 0 ]; then
    suites=("$@")
else
    suites=("${all_suites[@]}")
fi

failed=0
results=()
for suite in "${suites[@]}"; do
    script="$TE_PATH/qa/$suite/test_ascend.sh"
    echo "======================================================================"
    echo "[RUN] $suite"

    if [ ! -f "$script" ]; then
        echo "Ascend entry point not found: $script" >&2
        results+=("$suite: FAILED (missing script)")
        failed=1
        continue
    fi

    set +e
    TE_PATH="$TE_PATH" bash "$script"
    rc=$?
    set -e

    if [ "$rc" -ne 0 ]; then
        results+=("$suite: FAILED (exit $rc)")
        failed=1
        continue
    fi

    case "$suite" in
        L0_jax_distributed_unittest|L0_jax_unittest|L0_jax_wheel)
            results+=("$suite: SKIPPED (no JAX Ascend runtime)")
            ;;
        *)
            results+=("$suite: PASSED")
            ;;
    esac
done

echo "======================================================================"
echo "Ascend L0 summary"
for result in "${results[@]}"; do
    echo "  $result"
done

exit "$failed"
