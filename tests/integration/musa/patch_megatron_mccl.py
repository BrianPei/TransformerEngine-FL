#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Temporarily patch Megatron-LM-FL to accept the MUSA mccl backend."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <MCORE_PATH>")

    config = Path(sys.argv[1]) / "megatron/training/config/common_config.py"
    text = config.read_text()
    old = '    distributed_backend: Literal["nccl", "gloo"] = "nccl"\n'
    new = '    distributed_backend: Literal["nccl", "gloo", "mccl"] = "mccl"\n'
    if old not in text:
        raise SystemExit("expected distributed_backend definition not found")
    config.write_text(text.replace(old, new, 1))
    print("Patched Megatron distributed_backend to accept mccl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
