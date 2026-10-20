"""Exercise nightly completeness and failure handling without changing PR selection."""

from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

import generate_matrix
import nightly_report

ROOT = Path(__file__).resolve().parents[3]


class NightlyReportTests(unittest.TestCase):
    def setUp(self):
        self.env = {
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "2",
            "SOURCE_SHA": "a" * 40,
            "PLATFORM": "cuda",
            "EXPECTED_PLATFORMS": '["cuda", "ascend"]',
            "PLAN_RESULT": "success",
            "PIPELINE_RESULT": "success",
            "DOWNLOAD_RESULT": "success",
        }
        self.needs = {
            name: {"result": "success"}
            for name in (
                "checkout_and_config",
                "build_test_artifact",
                "unit_tests",
                "integration_tests",
                "all_tests_complete",
            )
        }
        self.needs["checkout_and_config"]["outputs"] = {
            "test_artifact_enabled": "true",
            "device_types": '["a100"]',
        }

    def report(self, platform="cuda"):
        return nightly_report.platform_report(
            dict(self.env, PLATFORM=platform, PLATFORM_NEEDS=json.dumps(self.needs))
        )

    def summary(self, reports, env=None):
        with tempfile.TemporaryDirectory() as directory:
            for index, report in enumerate(reports):
                Path(directory, f"report-{index}.json").write_text(json.dumps(report))
            return nightly_report.summarize(env or self.env, Path(directory))

    def test_required_jobs_must_succeed(self):
        for name, result in itertools.product(self.needs, ("failure", "cancelled", "skipped")):
            with self.subTest(name=name, result=result):
                needs = deepcopy(self.needs)
                needs[name]["result"] = result
                report = nightly_report.platform_report(
                    dict(self.env, PLATFORM_NEEDS=json.dumps(needs))
                )
                self.assertEqual(report["status"], "failed")
        self.assertEqual(self.report()["status"], "passed")

    def test_no_build_is_allowed_only_when_disabled(self):
        self.needs["build_test_artifact"]["result"] = "skipped"
        self.assertEqual(self.report()["status"], "failed")
        self.needs["checkout_and_config"]["outputs"]["test_artifact_enabled"] = "false"
        self.assertEqual(self.report()["status"], "passed")

    def test_complete_run_and_partial_rerun(self):
        reports = [self.report("cuda"), self.report("ascend")]
        self.assertEqual(self.summary(reports)["status"], "passed")
        reports[0]["attempt"] = "1"
        self.assertEqual(self.summary(reports)["status"], "passed")

    def test_missing_duplicate_failed_and_foreign_reports_fail(self):
        reports = [self.report("cuda"), self.report("ascend")]
        self.assertEqual(self.summary([])["status"], "failed")
        self.assertEqual(self.summary(reports[:1])["status"], "failed")
        self.assertEqual(self.summary(reports + reports[:1])["status"], "failed")
        for key, value in (
            ("status", "failed"),
            ("source_sha", "b" * 40),
            ("run_id", "456"),
            ("profile", "pr"),
            ("attempt", "3"),
            ("attempt", None),
            ("platform", "unknown"),
        ):
            with self.subTest(key=key, value=value):
                changed = deepcopy(reports)
                changed[0][key] = value
                self.assertEqual(self.summary(changed)["status"], "failed")

    def test_controller_and_matrix_failures_cannot_be_hidden_by_reports(self):
        reports = [self.report("cuda"), self.report("ascend")]
        for key, value in itertools.product(
            ("PLAN_RESULT", "PIPELINE_RESULT", "DOWNLOAD_RESULT"),
            ("failure", "cancelled", "skipped"),
        ):
            with self.subTest(key=key, value=value):
                self.assertEqual(
                    self.summary(reports, dict(self.env, **{key: value}))["status"], "failed"
                )

    def test_empty_plan_and_malformed_reports_fail(self):
        self.assertEqual(
            self.summary([], dict(self.env, EXPECTED_PLATFORMS="[]"))["status"], "failed"
        )
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "bad.json").write_text("{broken")
            self.assertEqual(
                nightly_report.summarize(self.env, Path(directory))["status"], "failed"
            )

    def test_cli_preserves_failure_summary_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "summary.json"
            markdown = root / "summary.md"
            command = [
                sys.executable,
                str(ROOT / ".github/scripts/ci/nightly_report.py"),
                "summary",
                "--input-dir",
                str(root / "missing-reports"),
                "--output",
                str(output),
            ]
            result = subprocess.run(
                command,
                env=dict(os.environ, **self.env, GITHUB_STEP_SUMMARY=str(markdown)),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads(output.read_text())["status"], "failed")
            self.assertIn("| cuda | missing |", markdown.read_text())


class NightlySelectionTests(unittest.TestCase):
    def test_nightly_selects_all_platforms_and_existing_full_matrices(self):
        configs = ROOT / ".github/configs"
        platforms = generate_matrix._load_watch_rules(configs)
        selected, _ = generate_matrix.select_platforms(platforms, ["README.md"], "schedule")
        self.assertEqual(selected, sorted(platforms))
        for platform in platforms:
            config = yaml.safe_load((configs / f"{platform}.yml").read_text())
            with self.subTest(platform=platform):
                self.assertTrue(config["integration_test_matrix"])
                self.assertTrue(config["unit_test_matrix"])
                self.assertNotIn("pr_unit_test_matrix", config)
                for group in config["unit_test_matrix"]:
                    self.assertNotIn("args", group)

    def test_nightly_controller_gate_and_profile_wiring(self):
        # BaseLoader preserves the YAML 'on' key rather than interpreting it as a bool.
        nightly = yaml.load(
            (ROOT / ".github/workflows/nightly.yml").read_text(), Loader=yaml.BaseLoader
        )
        ci = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
        self.assertNotIn("schedule", ci["on"])
        self.assertIn("schedule", nightly["on"])
        self.assertEqual(nightly["concurrency"]["cancel-in-progress"], "false")
        jobs = nightly["jobs"]
        self.assertEqual(jobs["complete"]["if"], "always()")
        self.assertEqual(jobs["platforms"]["strategy"]["fail-fast"], "false")
        self.assertEqual(jobs["platforms"]["with"]["test_profile"], "nightly")
        self.assertEqual(jobs["platforms"]["with"]["run_integration_tests"], "true")
        self.assertEqual(
            jobs["platforms"]["with"]["checkout_ref"], "${{ needs.plan.outputs.source_sha }}"
        )
        pr = ci["jobs"]["platform-pipeline"]["with"]
        self.assertEqual(pr, {"platform": "${{ matrix.platform }}"})
        pipeline = yaml.load(
            (ROOT / ".github/workflows/_platform_pipeline.yml").read_text(), Loader=yaml.BaseLoader
        )
        self.assertEqual(pipeline["jobs"]["test"]["with"], {"platform": "${{ inputs.platform }}"})
        unit = yaml.load(
            (ROOT / ".github/workflows/unit_tests_common.yml").read_text(), Loader=yaml.BaseLoader
        )
        self.assertNotIn("test_profile", unit["on"]["workflow_call"]["inputs"])
        regression = next(
            step
            for job in unit["jobs"].values()
            for step in job.get("steps", [])
            if step.get("name") == "Check coverage regression"
        )
        self.assertNotIn("test_profile", regression["if"])


if __name__ == "__main__":
    unittest.main()
