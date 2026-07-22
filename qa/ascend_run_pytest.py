#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Run pytest with Transformer Engine configured for Ascend/NPU."""

import os
import sys

os.environ.setdefault("PLATFORM", "ascend")
os.environ.setdefault("NVTE_DEVICE_TYPE", "npu")
os.environ.setdefault("TE_FL_SKIP_CUDA", "1")
os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")

import torch

try:
    import torch_npu  # noqa: F401  # pylint: disable=unused-import
except ImportError as exc:
    raise SystemExit(str(exc)) from exc

import transformer_engine

if transformer_engine.te_device_type() != "npu" or not torch.npu.is_available():
    raise SystemExit("Transformer Engine did not initialize an available Ascend NPU backend.")

import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
