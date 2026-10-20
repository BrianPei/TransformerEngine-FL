#!/usr/bin/env python3
"""Record platform job results and enforce completeness of a nightly run.

These reports exist independently of optional coverage collection. Workflow job
results cover the entire test matrices, including setup and coverage failures.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

SCHEMA = "te.nightly/v1"


def platform_report(env: dict[str, str]) -> dict:
    needs = json.loads(env["PLATFORM_NEEDS"])
    jobs = {name: value["result"] for name, value in needs.items()}
    outputs = needs["checkout_and_config"].get("outputs", {})
    required = ("checkout_and_config", "unit_tests", "integration_tests", "all_tests_complete")
    passed = all(jobs.get(name) == "success" for name in required)
    expected_build = "success" if outputs.get("test_artifact_enabled") == "true" else "skipped"
    passed = passed and jobs.get("build_test_artifact") == expected_build
    return {
        "schema_version": SCHEMA,
        "report_type": "platform",
        "profile": "nightly",
        "test_level": "L2",
        "run_id": env["GITHUB_RUN_ID"],
        "attempt": env["GITHUB_RUN_ATTEMPT"],
        "source_sha": env["SOURCE_SHA"],
        "platform": env["PLATFORM"],
        "status": "passed" if passed else "failed",
        "jobs": jobs,
        "devices": json.loads(outputs.get("device_types") or "[]"),
        "image_ref": outputs.get("ci_image", ""),
        "config_hash": outputs.get("config_hash", ""),
    }


def summarize(env: dict[str, str], directory: Path) -> dict:
    expected = json.loads(env["EXPECTED_PLATFORMS"])
    if not isinstance(expected, list) or any(not isinstance(p, str) or not p for p in expected):
        raise ValueError("Expected platforms must be an array of nonempty strings")
    errors = []
    if not expected or len(expected) != len(set(expected)):
        errors.append("Platform plan must be nonempty and unique")
    for name in ("PLAN_RESULT", "PIPELINE_RESULT", "DOWNLOAD_RESULT"):
        if env.get(name) != "success":
            errors.append(f"{name}: {env.get(name, 'missing')}")
    reports = {}
    for path in sorted(directory.rglob("*.json")):
        try:
            report = json.loads(path.read_text())
            if not isinstance(report, dict):
                raise ValueError("Report must be an object")
            platform = report.get("platform")
            if not isinstance(platform, str) or platform not in expected:
                raise ValueError(f"Unexpected platform: {platform}")
            if platform in reports:
                raise ValueError(f"Duplicate platform: {platform}")
            for key, value in (
                ("schema_version", SCHEMA),
                ("report_type", "platform"),
                ("profile", "nightly"),
                ("test_level", "L2"),
                ("source_sha", env["SOURCE_SHA"]),
                ("run_id", env["GITHUB_RUN_ID"]),
            ):
                if report.get(key) != value:
                    raise ValueError(f"{platform}: mismatched {key}")
            # Re-running failed jobs reuses artifacts from successful jobs in
            # earlier attempts of this same run and immutable source commit.
            if not 1 <= int(report.get("attempt", 0)) <= int(env["GITHUB_RUN_ATTEMPT"]):
                raise ValueError(f"{platform}: invalid attempt")
            reports[platform] = report
            if report.get("status") != "passed":
                errors.append(f"{platform}: {report.get('status', 'missing status')}")
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    for platform in sorted(set(expected) - reports.keys()):
        errors.append(f"Missing platform report: {platform}")
    return {
        "schema_version": SCHEMA,
        "report_type": "summary",
        "profile": "nightly",
        "test_level": "L2",
        "run_id": env["GITHUB_RUN_ID"],
        "attempt": env["GITHUB_RUN_ATTEMPT"],
        "source_sha": env["SOURCE_SHA"],
        "status": "failed" if errors else "passed",
        "expected_platforms": expected,
        "platforms": reports,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("platform", "summary"))
    parser.add_argument("--input-dir", type=Path, default=Path("nightly-reports"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ)
    report = platform_report(env) if args.mode == "platform" else summarize(env, args.input_dir)
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.mode == "summary" and env.get("GITHUB_STEP_SUMMARY"):
        lines = [
            f"### Nightly: {report['status']}",
            "",
            f"Source: `{report['source_sha']}` · Profile: nightly · Level: L2",
            "",
            "| Platform | Result | Config | Build | Unit | Integration |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for platform in report["expected_platforms"]:
            item = report["platforms"].get(platform, {})
            jobs = item.get("jobs", {})
            cells = [platform, item.get("status", "missing")]
            cells += [
                jobs.get(name, "missing")
                for name in (
                    "checkout_and_config",
                    "build_test_artifact",
                    "unit_tests",
                    "integration_tests",
                )
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(["", *[f"- {error}" for error in report["errors"]]])
        with Path(env["GITHUB_STEP_SUMMARY"]).open("a") as summary:
            summary.write("\n".join(lines) + "\n")
    # Platform jobs must upload failure reports before the caller's gate fails.
    return int(args.mode == "summary" and report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
