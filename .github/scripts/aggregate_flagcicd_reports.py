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
    parser.add_argument("--platform-job-result", default="success")
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

    # Check if test level requirements are met by report scopes (per platform)
    test_level = args.test_level
    if test_level in {"L2", "L3", "L4"}:
        # L2+ requires unit test reports from all platforms
        for platform in expected_platforms:
            platform_reports = [r for r in reports if r.get("platform", {}).get("name") == platform]
            platform_scopes = {
                r.get("metrics", {}).get("coverage", {}).get("scope", "") for r in platform_reports
            }
            if "unit" not in platform_scopes:
                errors.append(
                    f"test level {test_level} requires unit test report from platform: {platform}"
                )
        # Note: Integration tests currently don't produce structured reports.
        # The aggregated status only reflects unit test results. Integration
        # test failures are caught by the workflow-level gate in entry_complete,
        # which checks needs.run_tests.result. External systems should check
        # the workflow conclusion, not just this aggregated status field.

    # If platform jobs failed, override aggregated status
    final_status = "report_error" if errors else choose_status(statuses, missing_platforms)
    if args.platform_job_result not in {"success", "skipped"}:
        # Platform job failed or cancelled, reflect in aggregated status
        # Map workflow result to report status
        result_to_status = {
            "failure": "test_failed",
            "cancelled": "cancelled",
        }
        final_status = result_to_status.get(args.platform_job_result, "infra_error")

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
        "status": final_status,
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
        print("⚠️  Report aggregation completed with errors:")
        for error in errors:
            print(f"  - {error}")
        # Return non-zero to signal aggregation problems
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
