# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


pip3 install onnxruntime
pip3 install onnxruntime_extensions

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

pip3 install pytest==8.2.1 || error_exit "Failed to install pytest"
# NVTE_UnfusedDPA_Emulate_FP8=1 enables FP8 attention emulation when no native backend is available
if [ "${PLATFORM:-}" = "musa" ] || [ "${PLATFORM:-}" = "mthreads" ]; then
    ONNX_FILTER="test_export_layernorm_recipe or test_export_layernorm_zero_centered_gamma or test_export_layernorm_normalization or test_export_core_attention or test_export_ctx_manager"
    NVTE_UnfusedDPA_Emulate_FP8=1 python3 -m pytest -v --tb=auto \
        --junitxml=$XML_LOG_DIR/test_onnx_export.xml \
        $TE_PATH/tests/pytorch/test_onnx_export.py \
        -k "$ONNX_FILTER" \
        --no-header
else
    NVTE_UnfusedDPA_Emulate_FP8=1 python3 -m pytest --tb=auto \
        --junitxml=$XML_LOG_DIR/test_onnx_export.xml \
        $TE_PATH/tests/pytorch/test_onnx_export.py
fi
