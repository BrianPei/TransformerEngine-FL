# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest launcher that records the multi-process Ascend smoke test as JUnit."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


def test_two_npu_hccl_transformer_engine() -> None:
    if not torch.npu.is_available() or torch.npu.device_count() < 2:
        pytest.skip("At least two visible Ascend NPUs are required")

    worker = Path(__file__).with_name("ascend_hccl_distributed_worker.py")
    env = os.environ.copy()
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        pytest.fail("torchrun is not available in the active Python environment")
    command = [
        torchrun,
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        str(worker),
    ]
    subprocess.run(command, check=True, env=env)
