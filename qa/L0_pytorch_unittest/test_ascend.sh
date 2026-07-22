#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
: "${TE_PATH:=$REPO_ROOT}"
: "${XML_LOG_DIR:=$TE_PATH/logs/L0_pytorch_unittest-ascend}"
: "${PYTHON_BIN:=python3}"

mkdir -p "$XML_LOG_DIR"

export TE_FL_SKIP_CUDA=1
export NVTE_FRAMEWORK=pytorch

echo "[INFO] Running the shared real-NPU and plugin test suite for PyTorch on Ascend."
TE_PATH="$TE_PATH" XML_LOG_DIR="$XML_LOG_DIR" \
    bash "$TE_PATH/qa/L0_cppunittest/test_ascend.sh"

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

reason="The original PyTorch L0 suite is CUDA/NVIDIA-FP8 oriented; the Ascend path validates the supported real-NPU plugin backend and records unsupported CUDA paths as skipped."
write_skip "pytest_test_sanity.xml" "L0_pytorch_unittest_ascend.test_sanity" "$reason"
write_skip "pytest_test_recipe.xml" "L0_pytorch_unittest_ascend.test_recipe" "$reason"
write_skip "pytest_test_deferred_init.xml" "L0_pytorch_unittest_ascend.test_deferred_init" "$reason"
write_skip "pytest_test_numerics.xml" "L0_pytorch_unittest_ascend.test_numerics" "$reason"
write_skip "pytest_test_cuda_graphs.xml" "L0_pytorch_unittest_ascend.test_cuda_graphs" "$reason"
write_skip "pytest_test_jit.xml" "L0_pytorch_unittest_ascend.test_jit" "$reason"
write_skip "pytest_test_fused_rope.xml" "L0_pytorch_unittest_ascend.test_fused_rope" "$reason"
write_skip "pytest_test_nvfp4.xml" "L0_pytorch_unittest_ascend.test_nvfp4" "$reason"
write_skip "pytest_test_quantized_tensor.xml" "L0_pytorch_unittest_ascend.test_quantized_tensor" "$reason"
write_skip "pytest_test_float8blockwisetensor.xml" "L0_pytorch_unittest_ascend.test_float8blockwisetensor" "$reason"
write_skip "pytest_test_float8_blockwise_scaling_exact.xml" "L0_pytorch_unittest_ascend.test_float8_blockwise_scaling_exact" "$reason"
write_skip "pytest_test_float8_blockwise_gemm_exact.xml" "L0_pytorch_unittest_ascend.test_float8_blockwise_gemm_exact" "$reason"
write_skip "pytest_test_gqa.xml" "L0_pytorch_unittest_ascend.test_gqa" "$reason"
write_skip "pytest_test_fused_optimizer.xml" "L0_pytorch_unittest_ascend.test_fused_optimizer" "$reason"
write_skip "pytest_test_multi_tensor.xml" "L0_pytorch_unittest_ascend.test_multi_tensor" "$reason"
write_skip "pytest_test_fusible_ops.xml" "L0_pytorch_unittest_ascend.test_fusible_ops" "$reason"
write_skip "pytest_test_permutation.xml" "L0_pytorch_unittest_ascend.test_permutation" "$reason"
write_skip "pytest_test_parallel_cross_entropy.xml" "L0_pytorch_unittest_ascend.test_parallel_cross_entropy" "$reason"
write_skip "pytest_test_cpu_offloading.xml" "L0_pytorch_unittest_ascend.test_cpu_offloading" "$reason"
write_skip "pytest_test_cpu_offloading_v1.xml" "L0_pytorch_unittest_ascend.test_cpu_offloading_v1" "$reason"
write_skip "pytest_test_attention.xml" "L0_pytorch_unittest_ascend.test_attention" "$reason"
write_skip "pytest_test_kv_cache.xml" "L0_pytorch_unittest_ascend.test_kv_cache" "$reason"
write_skip "pytest_test_hf_integration.xml" "L0_pytorch_unittest_ascend.test_hf_integration" "$reason"
write_skip "pytest_test_checkpoint.xml" "L0_pytorch_unittest_ascend.test_checkpoint" "$reason"
