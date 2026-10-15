#!/usr/bin/env python3
"""Upload a report to FlagCICD using token authentication, without a user ID."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def upload(path: Path, file_type: str, job_name: str, backend_url: str) -> str:
    token = os.environ.get("FLAGCICD_API_TOKEN", "")
    if not path.is_file():
        return "Skipped: report file missing"
    if not token:
        return "Skipped: FLAGCICD_API_TOKEN unavailable"
    # Keep the token out of process arguments and never print response bodies,
    # which may contain server-provided credentials or reflected request data.
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "curl.conf"
        config.touch(mode=0o600)
        config.write_text("header = " + json.dumps(f"Authorization: Bearer {token}") + "\n")
        command = [
            "curl",
            "--config",
            str(config),
            "--noproxy",
            "*",
            "--silent",
            "--connect-timeout",
            "10",
            "--max-time",
            "60",
            "--output",
            os.devnull,
            "--write-out",
            "%{http_code}",
            "--form",
            f"file=@{path.resolve()}",
        ]
        fields = {
            "file_format": "json",
            "file_type": file_type,
            "is_zipped": "false",
            "git_project_name": os.environ.get("GITHUB_REPOSITORY", ""),
            "workflow_name": os.environ.get("GITHUB_WORKFLOW", ""),
            "job_name": job_name,
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "runner_name": os.environ.get("RUNNER_NAME", ""),
        }
        event_path = os.environ.get("GITHUB_EVENT_PATH")
        if event_path:
            event = json.loads(Path(event_path).read_text())
            pr_id = event.get("pull_request", {}).get("number")
            if not pr_id and event.get("issue", {}).get("pull_request"):
                pr_id = event["issue"].get("number")
            if pr_id:
                fields["pr_id"] = str(pr_id)
        for key, value in fields.items():
            command.extend(["--form-string", f"{key}={value}"])
        command.append(backend_url)
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=65)
    status = result.stdout.strip()
    if result.returncode:
        return f"Upload failed: curl exit {result.returncode}"
    if len(status) == 3 and status.isdigit() and 200 <= int(status) < 300:
        return f"HTTP {status}: accepted; verify report parsing in FlagCICD"
    return f"Upload failed: HTTP {status if status.isdigit() else 'unavailable'}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--file-type", required=True)
    parser.add_argument("--backend-url", default="http://flagcicd-inner.flagos.net:8000/metrics/")
    args = parser.parse_args()
    job_name = args.report.stem
    try:
        result = upload(args.report, args.file_type, job_name, args.backend_url)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result = f"Upload failed: {type(error).__name__}; check uploader configuration/connectivity"
    if not result.startswith("HTTP "):
        print(f"::warning::{result}")
    else:
        print(result)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"FlagCICD {job_name}: {result}\n")
    # Report delivery must not override test results.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
