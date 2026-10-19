#!/usr/bin/env python3
"""Aggregate platform FlagCICD reports without merging coverage denominators."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "flagcicd.report/v1"
STATUS_PRIORITY = {
    "report_error": 5,
    "infra_error": 4,
    "build_failed": 4,
    "test_failed": 3,
    "cancelled": 2,
    "skipped": 1,
    "unsupported": 1,
    "passed": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def choose_status(statuses: list[str], missing_platforms: set[str]) -> str:
    if missing_platforms:
        return "infra_error"
    return max(
        statuses,
        key=lambda status: STATUS_PRIORITY.get(status, 4),
        default="infra_error",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-platforms", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--parent-run-id", default="")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--test-level", required=True)
    parser.add_argument("--risk-level", required=True)
    args = parser.parse_args()

    expected_platforms = json.loads(args.expected_platforms)
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(args.input_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        if payload.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{path}: unexpected schema_version")
            continue
        reports.append(payload)

    statuses = [report.get("status", "report_error") for report in reports]
    actual_platforms = {report.get("platform", {}).get("name", "") for report in reports}
    missing_platforms = set(expected_platforms) - actual_platforms
    if missing_platforms:
        errors.append(f"missing platform reports: {sorted(missing_platforms)}")
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_type": "run_summary",
        "run_id": args.run_id,
        "task_id": args.task_id,
        "parent_run_id": args.parent_run_id,
        "repository": "TransformerEngine-FL",
        "source_sha": args.source_sha,
        "test_level": args.test_level,
        "risk_level": args.risk_level,
        "status": choose_status(statuses, missing_platforms),
        "platform": {"name": "multi", "device": "multi"},
        "metrics": {
            "coverage": {
                "available": False,
                "aggregation": "per_platform_only",
                "children": [
                    {
                        "platform": report.get("platform", {}).get("name", ""),
                        "device": report.get("platform", {}).get("device", ""),
                        "coverage": report.get("metrics", {}).get("coverage", {}),
                    }
                    for report in reports
                ],
            }
        },
        "children": reports,
        "artifacts": [
            {"kind": "child_report", "path": str(path)}
            for path in sorted(args.input_dir.rglob("*.json"))
        ],
        "errors": errors,
        "created_at": utc_now(),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Created FlagCICD run summary: {args.output}")
    if errors:
        print("Report aggregation completed with errors:")
        for error in errors:
            print(f"- {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
