# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Write a one-case JUnit report for an explicitly unsupported Ascend QA suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suites = ElementTree.Element(
        "testsuites",
        {"name": args.suite, "tests": "1", "failures": "0", "errors": "0", "skipped": "1"},
    )
    suite = ElementTree.SubElement(
        suites,
        "testsuite",
        {"name": args.suite, "tests": "1", "failures": "0", "errors": "0", "skipped": "1"},
    )
    case = ElementTree.SubElement(
        suite,
        "testcase",
        {"classname": args.suite, "name": "ascend_runtime_support"},
    )
    ElementTree.SubElement(case, "skipped", {"message": args.reason}).text = args.reason
    ElementTree.indent(suites)
    ElementTree.ElementTree(suites).write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
