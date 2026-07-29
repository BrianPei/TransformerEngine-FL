#!/usr/bin/env bash

# Hygon/DTK workflow configuration.  Add or remove entries here as backend
# support changes; keep the generic runner logic in test.sh branch-free.

HYGON_L0_SKIP_LABELS=(
    "test_sanity.py"
    "test_recipe.py"
    "test_numerics.py"
    "test_cuda_graphs.py"
    "test_fused_rope.py"
    "test_quantized_tensor.py"
    "test_float8blockwisetensor.py"
    "test_float8_blockwise_scaling_exact.py"
    "test_float8_blockwise_gemm_exact.py"
    "test_gqa.py"
    "test_fused_optimizer.py"
    "test_multi_tensor.py"
    "test_fusible_ops.py"
    "test_permutation.py"
    "test_parallel_cross_entropy.py"
    "test_cpu_offloading.py"
    "test_cpu_offloading_v1.py"
    "test_attention.py"
    "test_kv_cache.py"
    "test_hf_integration.py"
    "test_checkpoint.py"
)

HYGON_DISTRIBUTED_SKIP_LABELS=(
    "test_numerics.py"
    "test_numerics_exact.py"
    "test_torch_fsdp2.py"
    "test_cast_master_weights_to_fp8.py"
    "test_numerics.py (debug)"
)

HYGON_ONNX_SKIP_GROUPS=(
    "test_export_linear"
    "test_export_layernorm_linear"
    "test_export_layernorm_mlp"
    "test_export_core_attention"
    "test_export_transformer_layer"
    "test_export_multihead_attention"
    "test_export_gpt_generation"
)
