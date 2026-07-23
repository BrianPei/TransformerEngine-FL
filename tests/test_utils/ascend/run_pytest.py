#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Run pytest with Transformer Engine configured for Ascend/NPU."""

import os
import sys

os.environ.setdefault("PLATFORM", "ascend")
os.environ.setdefault("TE_FL_SKIP_CUDA", "1")
os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from npu_patch import apply_ascend_npu_patch

try:
    apply_ascend_npu_patch()
except RuntimeError as exc:
    raise SystemExit(str(exc)) from exc

import pytest

raise SystemExit(pytest.main(sys.argv[1:]))
