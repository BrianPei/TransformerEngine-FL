#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Run pytest with the Ascend backend compatibility layer enabled."""

from __future__ import annotations

import sys

from adapter import apply


def main(argv: list[str] | None = None) -> int:
    # The adapter must run before pytest imports and collects the selected
    # tests, because some upstream tests import CUDA-oriented helpers at
    # module load time.
    apply()

    import pytest

    return pytest.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main())
