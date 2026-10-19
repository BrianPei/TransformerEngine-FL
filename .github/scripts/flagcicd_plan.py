#!/usr/bin/env python3
"""Resolve a TE-FL FlagCICD level, risk, and artifact identity."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any


LEVEL_ORDER = {f"L{index}": index for index in range(5)}
RISK_ORDER = {f"R{index}": index for index in range(1, 5)}

RULES = (
    {
        "id": "native-extension",
        "patterns": (
            "transformer_engine/**/csrc/**",
            "transformer_engine/**/cpp/**",
            "transformer_engine/**/cuda/**",
        ),
        "minimum_level": "L3",
        "risk": "R3",
    },
    {
        "id": "ci-and-build",
        "patterns": (
            ".github/**",
            "CMakeLists.txt",
            "cmake/**",
            "setup.py",
            "pyproject.toml",
        ),
        "minimum_level": "L2",
        "risk": "R2",
    },
    {
        "id": "numerical-and-performance",
        "patterns": (
            "tests/pytorch/**",
            "tests/plugin/**",
            "qa/**",
        ),
        "minimum_level": "L1",
        "risk": "R1",
    },
)


def matches(path: str, pattern: str) -> bool:
    """Match repository-relative paths with a small, CI-portable glob."""

    # Remove exactly "./" prefix if present, don't use lstrip which removes chars
    if path.startswith("./"):
        normalized = path[2:]
    else:
        normalized = path
    if fnmatch.fnmatchcase(normalized, pattern):
        return True
    if pattern.endswith("/**") and normalized.startswith(pattern[:-3]):
        return True
    return False


def max_level(*levels: str) -> str:
    return max(levels, key=lambda value: LEVEL_ORDER[value])


def max_risk(*risks: str) -> str:
    return max(risks, key=lambda value: RISK_ORDER[value])


def resolve(args: argparse.Namespace) -> dict[str, Any]:
    changed_files = json.loads(args.changed_files)
    if not isinstance(changed_files, list) or not all(
        isinstance(path, str) for path in changed_files
    ):
        raise ValueError("changed-files must be a JSON array of strings")

    requested_level = args.requested_level
    if requested_level == "auto":
        requested_level = "L1"
    if requested_level not in LEVEL_ORDER:
        raise ValueError(f"unsupported requested level: {requested_level}")

    requested_risk = args.requested_risk
    if requested_risk == "auto":
        requested_risk = "R1"
    if requested_risk not in RISK_ORDER:
        raise ValueError(f"unsupported requested risk: {requested_risk}")

    matched_rules: list[dict[str, str]] = []
    unknown_files: list[str] = []
    inferred_level = "L0"
    inferred_risk = "R1"
    for path in changed_files:
        file_rules = [
            rule for rule in RULES if any(matches(path, pattern) for pattern in rule["patterns"])
        ]
        if not file_rules:
            unknown_files.append(path)
            inferred_level = max_level(inferred_level, "L2")
            inferred_risk = max_risk(inferred_risk, "R2")
            continue
        for rule in file_rules:
            matched_rules.append({"path": path, "rule": rule["id"]})
            inferred_level = max_level(inferred_level, rule["minimum_level"])
            inferred_risk = max_risk(inferred_risk, rule["risk"])

    # An empty changed-file list is a manual run without a diff. It keeps the
    # requested level; an explicit unknown path escalates to L2/R2.
    effective_level = max_level(requested_level, inferred_level)
    effective_risk = max_risk(requested_risk, inferred_risk)
    source_sha = args.source_sha or "unknown"
    config_hash = args.config_hash or "unknown"
    source_key = source_sha[:12]
    config_key = config_hash[:12]
    platform = args.platform or "multi"
    artifact_key = f"test-artifact-{platform}-{source_key}-{config_key}"
    return {
        "schema_version": "flagcicd.plan/v1",
        "requested_level": requested_level,
        "requested_risk": requested_risk,
        "test_level": effective_level,
        "risk_level": effective_risk,
        "risk_rule_version": "v1",
        "changed_files": changed_files,
        "matched_rules": matched_rules,
        "unknown_files": unknown_files,
        "artifact_key": artifact_key,
        "platform": platform,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--changed-files", default="[]")
    parser.add_argument("--requested-level", default="L1")
    parser.add_argument("--requested-risk", default="R1")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--config-hash", default="")
    parser.add_argument("--platform", default="multi")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = resolve(args)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Created FlagCICD plan: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
