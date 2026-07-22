# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os

import transformer_engine
import transformer_engine.pytorch

requested_device = os.getenv("NVTE_DEVICE_TYPE", "").strip().lower()
if requested_device in ("ascend", "npu"):
    assert transformer_engine.te_device_type() == "npu"

print("OK")
