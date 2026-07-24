#!/usr/bin/env python3
"""Run unmodified upstream Transformer Engine tests with an optional adapter."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def _load_adapter(adapter_path: Path) -> None:
    sys.path.insert(0, str(adapter_path.parent))
    spec = importlib.util.spec_from_file_location("te_test_platform_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load test adapter: {adapter_path}")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    apply = getattr(adapter, "apply", None)
    if not callable(apply):
        raise RuntimeError(f"Test adapter must define apply(): {adapter_path}")
    apply()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--adapter", type=Path)
    args, pytest_args = parser.parse_known_args(argv)

    if args.adapter is not None:
        _load_adapter(args.adapter.resolve())

    import pytest

    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
