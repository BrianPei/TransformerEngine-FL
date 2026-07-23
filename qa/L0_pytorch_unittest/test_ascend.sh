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

FAIL=0

run_test_step() {
    local xml_file=$1
    local test_path=$2
    shift 2
    local runner=pytest
    local label
    label=$(basename "$test_path")
    local command=("$PYTHON_BIN" -m pytest)
    local target_args=("$test_path")

    if [ "$#" -gt 0 ]; then
        runner=${1:-pytest}
        shift
    fi
    if [ "$#" -gt 0 ]; then
        label=${1:-$(basename "$test_path")}
        shift
    fi
    if [ "$#" -gt 0 ]; then
        target_args=("$@")
    fi

    if [ "$runner" = "ascend" ]; then
        command=("$PYTHON_BIN" "$TE_PATH/qa/ascend_run_pytest.py")
    fi

    echo "-------------------------------------------------------"
    echo "[RUN] Executing: $label"
    "${command[@]}" \
        -v -s --tb=short \
        --junitxml="$XML_LOG_DIR/$xml_file" \
        "${target_args[@]}" || FAIL=1
}

echo "[INFO] Running real-NPU TE unit tests on Ascend."
run_test_step \
    "pytest_ascend_backend_smoke.xml" \
    "$TE_PATH/qa/test_backend_ascend_smoke.py" \
    ascend \
    "Ascend TE module smoke tests"
run_test_step \
    "pytest_ascend_backend_ops.xml" \
    "$TE_PATH/qa/test_backend_ascend_ops.py" \
    ascend \
    "Ascend FlagOS operator tests"

echo "[INFO] Running portable Plugin Unit files shared with CUDA and MetaX."
PLUGIN_TEST_ROOT="$TE_PATH/transformer_engine/plugin/tests"
run_test_step "pytest_test_plugin_policy.xml" \
    "$PLUGIN_TEST_ROOT/test_plugin_policy.py"
run_test_step "pytest_test_plugin_manager.xml" \
    "$PLUGIN_TEST_ROOT/test_plugin_manager.py"
run_test_step "pytest_test_backend_flagos.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_flagos.py"

# The FlagOS implementation lifecycle files use CPU tensors and import-time
# flag_gems mocks. Real Ascend GEMM, RMSNorm, softmax, and multi-tensor paths
# are covered by qa/test_backend_ascend_ops.py above. Fused Adam is not in the
# current Ascend-supported Unit scope.
run_test_step "pytest_test_backend_reference.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_reference.py"
run_test_step "pytest_test_backend_reference_activation.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_reference_activation.py"
run_test_step "pytest_test_backend_reference_dropout.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_reference_dropout.py"
run_test_step "pytest_test_backend_reference_gemm.xml" \
    "$PLUGIN_TEST_ROOT/test_backend_reference_gemm.py"

echo "[INFO] Running selected shared PyTorch sanity and numerics tests on Ascend."
NVTE_FLASH_ATTN=0 \
NVTE_FUSED_ATTN=0 \
NVTE_UNFUSED_ATTN=1 \
run_test_step \
    "pytest_shared_sanity_portable.xml" \
    "$TE_PATH/tests/pytorch/test_sanity.py" \
    ascend \
    "Shared portable sanity tests" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_normalization_amp[LayerNorm-False-False-small-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_normalization_amp[RMSNorm-False-False-small-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_linear[False-False-False-small-None-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_layernorm_linear[False-LayerNorm-False-False-False-small-None-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_layernorm_linear[False-RMSNorm-False-False-False-small-None-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_layernorm_mlp[False-False-LayerNorm-gelu-False-False-False-small-None-dtype0]" \
    "$TE_PATH/tests/pytorch/test_sanity.py::test_sanity_layernorm_mlp[False-False-RMSNorm-silu-False-False-False-small-None-dtype0]"

NVTE_FLASH_ATTN=0 \
NVTE_FUSED_ATTN=0 \
NVTE_UNFUSED_ATTN=1 \
run_test_step \
    "pytest_shared_numerics_portable.xml" \
    "$TE_PATH/tests/pytorch/test_numerics.py" \
    ascend \
    "Shared non-FP8 numerics and unfused attention tests" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_linear_accuracy[False-False-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_linear_accuracy[False-False-small-1-dtype1]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_accuracy[False-1e-05-126m-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_rmsnorm_accuracy[False-1e-05-126m-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_linear_accuracy[False-False-False-LayerNorm-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_linear_accuracy[False-False-False-RMSNorm-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_mlp_accuracy[False-False-LayerNorm-gelu-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_layernorm_mlp_accuracy[False-False-RMSNorm-silu-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_dpa_accuracy[126m-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_mha_accuracy[causal-small-1-dtype0]" \
    "$TE_PATH/tests/pytorch/test_numerics.py::test_mha_accuracy[no_mask-small-1-dtype0]"

if [ "$FAIL" -ne 0 ]; then
    echo "Some Ascend PyTorch Unit tests failed."
    exit 1
fi

echo "All assigned Ascend PyTorch Unit tests passed (some might have been skipped)."
