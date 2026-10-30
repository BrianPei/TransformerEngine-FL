#!/usr/bin/env bash
# Iluvatar backend test configuration, matching the other platform adapters.
set -euo pipefail

readonly PLATFORM_NAME="iluvatar"
readonly PLATFORM_DISPLAY_NAME="Iluvatar BI-V150"
readonly PLATFORM_TYPE="shell_launcher"
readonly PLATFORM_NPROC_PER_NODE=8
readonly PLATFORM_DEVICE_ENV_VAR="CUDA_VISIBLE_DEVICES"
readonly PLATFORM_UNIT_TIMEOUT=14400
readonly PLATFORM_INTEGRATION_TIMEOUT=1800
readonly PLATFORM_UNIT_TEST_PATHS=(
    "tests/plugin/backend/iluvatar"
    "tests/plugin/backend/reference"
    "tests/plugin/backend/flagos"
)
readonly PLATFORM_INTEGRATION_TEST_SCRIPT="tests/plugin/backend/iluvatar/integration_test.sh"
readonly PLATFORM_PYTEST_UNIT_MARKERS="not slow and not integration"
readonly PLATFORM_PYTEST_INTEGRATION_MARKERS="integration"
readonly PLATFORM_PYTEST_EXTRA_ARGS="--tb=short --verbose"
readonly PLATFORM_PYTEST_DESELECT=""
readonly PLATFORM_COVERAGE_ENABLED=true

# Whole-file exclusions for this CI baseline, not passing or capability-skipped
# tests. Keep the shared sources intact; restore these entries after isolated
# reproduction confirms the attention/graph/offload issues are fixed.
readonly ILUVATAR_EXCLUDED_TEST_TARGETS=(
    "tests/pytorch/test_numerics.py|FP16 GPT CUDA Graph case hung with a driver thread remaining after termination."
    "tests/pytorch/test_cuda_graphs.py|FP16 Transformer graph case hung again on a different GPU."
    "tests/pytorch/test_cpu_offloading.py|UnfusedAttention MHA numerics produced no output for 20 minutes and left an unreaped process."
    "tests/pytorch/test_cpu_offloading_v1.py|MHA/Transformer offload cases require an unavailable fused attention backend; excluded with the offload suite."
)

# Common L0 QA targets plus plugin tests. Match PPU's RoPE coverage through
# tests/plugin/backend/flagos/test_fused_rope.py (included by the FlagOS directory
# below), rather than the large tests/pytorch/test_fused_rope.py matrix. On the
# 32 GiB BI-V150 that matrix OOMs in gradient comparisons and cascades across
# cases; it remains available for separate capacity testing.
readonly ILUVATAR_UNIT_TEST_TARGETS=(
    "tests/pytorch/test_sanity.py"
    "tests/pytorch/test_recipe.py"
    "tests/pytorch/test_deferred_init.py"
    "tests/pytorch/test_jit.py"
    "tests/pytorch/nvfp4"
    "tests/pytorch/test_quantized_tensor.py"
    "tests/pytorch/test_float8blockwisetensor.py"
    "tests/pytorch/test_float8_blockwise_scaling_exact.py"
    "tests/pytorch/test_float8_blockwise_gemm_exact.py"
    "tests/pytorch/test_gqa.py"
    "tests/pytorch/test_fused_optimizer.py"
    "tests/pytorch/test_multi_tensor.py"
    "tests/pytorch/test_fusible_ops.py"
    "tests/pytorch/test_permutation.py"
    "tests/pytorch/test_parallel_cross_entropy.py"
    "tests/pytorch/attention/test_attention.py"
    "tests/pytorch/attention/test_kv_cache.py"
    "tests/pytorch/test_hf_integration.py"
    "tests/pytorch/test_checkpoint.py"
    "tests/plugin/plugin"
    "tests/plugin/backend/reference"
    "tests/plugin/backend/flagos"
)
readonly ILUVATAR_DISTRIBUTED_TEST_TARGETS=(
    "tests/pytorch/distributed/test_numerics.py"
    "tests/pytorch/distributed/test_numerics_exact.py"
    "tests/pytorch/distributed/test_torch_fsdp2.py"
    "tests/pytorch/attention/test_cp_utils.py"
    "tests/pytorch/distributed/test_cast_master_weights_to_fp8.py"
)
