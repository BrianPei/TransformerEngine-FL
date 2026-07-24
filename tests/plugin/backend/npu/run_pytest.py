#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compatibility entry point for running upstream TE tests on Ascend/NPU."""

import os
from pathlib import Path
import sys

os.environ.setdefault("PLATFORM", "ascend")
os.environ.setdefault("TE_FL_SKIP_CUDA", "1")
os.environ.setdefault("NVTE_FRAMEWORK", "pytorch")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

PLUGIN_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PLUGIN_TEST_ROOT))

from run_upstream import main


raise SystemExit(
    main(
        [
            "--adapter",
            str(Path(__file__).with_name("adapter.py")),
            *sys.argv[1:],
        ]
    )
)
