# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Runtime patches for running Transformer Engine pytest suites on Ascend NPU."""

from __future__ import annotations

import os


def _set_ascend_env() -> None:
    os.environ.setdefault("PLATFORM", "ascend")
    os.environ.setdefault("TE_FL_SKIP_CUDA", "1")
    os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def _unsupported(reason: str):
    return False, reason


def apply_ascend_npu_patch() -> None:
    """Configure TE and patch CUDA-only helpers for Ascend test execution."""
    _set_ascend_env()

    import torch

    try:
        import torch_npu
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"torch_npu is required for Ascend tests: {exc}") from exc

    import transformer_engine

    transformer_engine.TE_DEVICE_TYPE = "npu"
    transformer_engine.TE_PLATFORM = torch_npu.npu

    # Some TE PyTorch paths query CUDA graph state unconditionally.
    torch.cuda.current_device = lambda: 0
    torch.cuda.is_current_stream_capturing = lambda: False

    _patch_quantization_capability_checks()
    _patch_te_gemm_workspace()


def _patch_quantization_capability_checks() -> None:
    import transformer_engine.pytorch.quantization as quantization

    quantization.check_fp8_support = lambda: _unsupported(
        "FP8 execution is not supported on npu."
    )
    quantization.check_mxfp8_support = lambda: _unsupported(
        "MXFP8 execution is not supported on npu."
    )
    quantization.check_nvfp4_support = lambda: _unsupported(
        "NVFP4 execution is not supported on npu."
    )
    quantization.check_fp8_block_scaling_support = lambda: _unsupported(
        "FP8 block scaling is not supported on npu."
    )


def _patch_te_gemm_workspace() -> None:
    import torch
    import transformer_engine.pytorch.cpp_extensions.gemm as gemm

    def _npu_workspace(device, ub, grouped_gemm):
        device_index = torch.npu.current_device() if device is None else int(device)
        npu_device = torch.device("npu", device_index)
        workspace_size = 4_194_304

        if ub:
            return torch.empty(workspace_size * 3, dtype=torch.uint8, device=npu_device)
        if grouped_gemm:
            return [torch.empty(workspace_size, dtype=torch.uint8, device=npu_device)]
        return torch.empty(workspace_size, dtype=torch.uint8, device=npu_device)

    cache_clear = getattr(gemm.get_cublas_workspace, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    gemm.get_cublas_workspace = _npu_workspace
