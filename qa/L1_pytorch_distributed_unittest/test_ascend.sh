#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L1_pytorch_distributed_unittest-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"
export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch

"$PYTHON_BIN" "$TE_PATH/qa/ascend_run_pytest.py" -v -s \
    --junitxml="$XML_LOG_DIR/pytest_hccl_te.xml" \
    "$TE_PATH/qa/test_ascend_hccl_distributed.py"

write_skip() {
    local xml_file=$1
    local suite=$2
    local reason=$3

    "$PYTHON_BIN" "$TE_PATH/qa/ascend_write_junit_skip.py" \
        --output "$XML_LOG_DIR/$xml_file" \
        --suite "$suite" \
        --reason "$reason"
    echo "[SKIP] $suite: $reason"
}

reason="The original distributed suite targets CUDA/NCCL, FP8 numerics, FSDP, and debug numerics paths; the Ascend path validates the supported HCCL two-rank TE smoke instead."
write_skip "pytest_test_numerics.xml" "L1_pytorch_distributed_unittest_ascend.test_numerics" "$reason"
write_skip "pytest_test_numerics_exact.xml" "L1_pytorch_distributed_unittest_ascend.test_numerics_exact" "$reason"
write_skip "pytest_test_torch_fsdp2.xml" "L1_pytorch_distributed_unittest_ascend.test_torch_fsdp2" "$reason"
write_skip "pytest_test_cp_utils.xml" "L1_pytorch_distributed_unittest_ascend.test_cp_utils" "$reason"
write_skip "pytest_test_cast_master_weights_to_fp8.xml" "L1_pytorch_distributed_unittest_ascend.test_cast_master_weights_to_fp8" "$reason"
write_skip "pytest_test_numerics_2.xml" "L1_pytorch_distributed_unittest_ascend.test_numerics_debug" "$reason"
