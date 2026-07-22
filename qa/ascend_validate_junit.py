#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Validate that an Ascend pytest job executed a meaningful number of tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def _suite_totals(path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return tests, failures, errors, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_files", nargs="+", type=Path)
    parser.add_argument("--min-tests", type=int, required=True)
    parser.add_argument("--min-passed", type=int, required=True)
    parser.add_argument("--max-skipped", type=int)
    args = parser.parse_args()

    totals = [sum(values) for values in zip(*(_suite_totals(path) for path in args.xml_files))]
    tests, failures, errors, skipped = totals
    passed = tests - failures - errors - skipped
    print(
        "Ascend JUnit summary: "
        f"tests={tests} passed={passed} skipped={skipped} failures={failures} errors={errors}"
    )

    if failures or errors:
        raise SystemExit("JUnit contains failed or errored tests")
    if tests < args.min_tests:
        raise SystemExit(f"Expected at least {args.min_tests} collected tests, got {tests}")
    if passed < args.min_passed:
        raise SystemExit(f"Expected at least {args.min_passed} passed tests, got {passed}")
    if args.max_skipped is not None and skipped > args.max_skipped:
        raise SystemExit(f"Expected at most {args.max_skipped} skipped tests, got {skipped}")


if __name__ == "__main__":
    main()
