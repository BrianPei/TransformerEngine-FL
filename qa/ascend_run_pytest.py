#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Run pytest with Transformer Engine configured for Ascend/NPU."""

import os
import sys

os.environ.setdefault("PLATFORM", "ascend")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

try:
    import torch_npu
except ModuleNotFoundError as exc:
    raise SystemExit(f"torch_npu is required for Ascend pytest: {exc}") from exc

import transformer_engine

transformer_engine.TE_DEVICE_TYPE = "npu"
transformer_engine.TE_PLATFORM = torch_npu.npu

# Some TE PyTorch paths call this single CUDA graph query unconditionally.
torch.cuda.is_current_stream_capturing = lambda: False

import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
