"""Regression checks for platform routing and the actual workflow gate commands.

Run without importing TransformerEngine or requiring accelerator hardware:
    python -m unittest discover -s .github/scripts/ci -p 'test_*.py'
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import yaml

import generate_matrix as matrix


ROOT = Path(__file__).resolve().parents[3]
STATUSES = ("success", "failure", "cancelled", "skipped")


class PlatformRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = matrix._load_watch_rules(ROOT / ".github/configs")
        cls.platforms = sorted(cls.rules)

    def select(self, *paths, event="pull_request"):
        return matrix.select_platforms(self.rules, list(paths), event)[0]

    def test_platform_discovery_excludes_metadata(self):
        self.assertTrue(self.platforms, "At least one platform configuration is required")
        self.assertTrue(matrix._NON_PLATFORM_STEMS.isdisjoint(self.platforms))

    def test_each_configured_watch_path_selects_its_platform(self):
        for platform, watch in self.rules.items():
            with self.subTest(platform=platform):
                self.assertTrue(watch, f"Declare paths_watch for {platform}")
                self.assertIn(f".github/configs/{platform}.yml", watch)
            for entry in watch:
                # Probe directory ownership without depending on a particular
                # test-script name or requiring every platform to share a layout.
                path = entry + "routing_probe.py" if entry.endswith("/") else entry
                with self.subTest(platform=platform, path=path):
                    self.assertEqual(self.select(path), [platform])

    def test_platform_addition_and_removal_are_config_driven(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            for platform in ("existing", "new-accelerator"):
                (configs / f"{platform}.yml").write_text(
                    f"paths_watch:\n  - .github/configs/{platform}.yml\n"
                    f"  - tests/plugin/backend/{platform}/\n"
                )
            # Metadata must be ignored before attempting to parse it as a
            # platform config, regardless of its contents.
            for stem in matrix._NON_PLATFORM_STEMS:
                (configs / f"{stem}.yml").write_text("[]\n")
            rules = matrix._load_watch_rules(configs)
            self.assertEqual(set(rules), {"existing", "new-accelerator"})
            self.assertEqual(
                matrix.select_platforms(
                    rules, ["tests/plugin/backend/new-accelerator/test.py"], "pull_request"
                )[0],
                ["new-accelerator"],
            )
            (configs / "existing.yml").unlink()
            rules = matrix._load_watch_rules(configs)
            self.assertEqual(set(rules), {"new-accelerator"})

    def test_metadata_only_directory_is_not_a_platform_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory)
            (configs / "template.yml").write_text("paths_watch: []\n")
            with self.assertRaisesRegex(ValueError, "No platform configurations"):
                matrix._load_watch_rules(configs)

    def test_vendor_aliases(self):
        for vendor, platform in (("npu", "ascend"), ("kunlunxin", "kunlun")):
            self.assertEqual(
                self.select(f"transformer_engine/plugin/core/backends/vendor/{vendor}/ops.py"),
                [platform],
            )

    def test_multiple_platforms_and_documentation(self):
        self.assertEqual(
            self.select(
                "tests/plugin/backend/ascend/test.py",
                "tests/plugin/backend/cuda/test.py",
                "README.md",
            ),
            ["ascend", "cuda"],
        )

    def test_shared_or_unknown_path_in_mixed_pr_runs_all(self):
        for path in (
            "transformer_engine/pytorch/attention.py",
            "transformer_engine/plugin/core/registry.py",
            "transformer_engine/plugin/core/backends/reference/impl/gemm.py",
            "transformer_engine/plugin/core/backends/flagos/flagos.py",
            ".github/workflows/all_tests_common.yml",
            ".github/scripts/ci/generate_matrix.py",
            "tests/test_utils/run_ci_test_group.py",
            "examples/pytorch/new_example.py",
            "setup.py",
            "unknown/new.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    self.select("tests/plugin/backend/cuda/test.py", path), self.platforms
                )

    def test_documentation_only(self):
        self.assertEqual(self.select("README.rst", "docs/conf.py", "examples/README.md"), [])

    def test_missing_diff_and_non_pr_events_run_all(self):
        self.assertEqual(self.select(), self.platforms)
        for event in ("push", "schedule", "workflow_dispatch"):
            self.assertEqual(self.select("README.md", event=event), self.platforms)

    def test_path_boundaries(self):
        self.assertFalse(
            matrix._matches("other/.github/configs/cuda.yml", ".github/configs/cuda.yml")
        )
        self.assertFalse(matrix._matches("tests/cuda_extra/test.py", "tests/cuda/"))

    def test_missing_watch_rules_run_all(self):
        rules = dict(self.rules, cuda=[])
        self.assertEqual(
            matrix.select_platforms(rules, ["tests/plugin/backend/ascend/test.py"], "pull_request")[
                0
            ],
            self.platforms,
        )

    def test_invalid_configs_fail(self):
        for content in ("[]", "paths_watch: cuda", "paths_watch: [null]", "invalid: ["):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                Path(directory, "cuda.yml").write_text(content)
                with self.assertRaises((ValueError, yaml.YAMLError)):
                    matrix._load_watch_rules(Path(directory))

    def test_rename_keeps_source_and_destination(self):
        with patch.object(matrix.subprocess, "run") as run:
            run.return_value.stdout = (
                "tests/plugin/backend/cuda/test.py\0tests/plugin/backend/ascend/test.py\0"
            )
            paths = matrix._changed_files("main")
            self.assertIn("--no-renames", run.call_args.args[0])
            self.assertEqual(self.select(*paths), ["ascend", "cuda"])


class WorkflowGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
        cls.common = yaml.safe_load((ROOT / ".github/workflows/all_tests_common.yml").read_text())[
            "jobs"
        ]

    def test_top_level_gate_status_combinations(self):
        gate = self.ci["ci-complete"]
        self.assertEqual(gate["if"], "always()")
        self.assertEqual(set(gate["needs"]), {"lint", "generate-matrix", "platform-pipeline"})
        command = gate["steps"][0]["run"]
        cases = [
            (lint, generated, pipeline, platforms)
            for lint, generated, pipeline in itertools.product(STATUSES, repeat=3)
            for platforms in ('["cuda"]', "[]")
        ]
        cases += [("success", "success", "success", value) for value in ("", "null", "{}", "[1]")]
        for lint, generated, pipeline, platforms in cases:
            expected = lint == generated == "success" and (
                (platforms == '["cuda"]' and pipeline == "success")
                or (platforms == "[]" and pipeline == "skipped")
            )
            with self.subTest(
                lint=lint, generated=generated, pipeline=pipeline, platforms=platforms
            ):
                result = subprocess.run(
                    ["bash", "-e", "-c", command],
                    env=dict(
                        os.environ,
                        LINT_RESULT=lint,
                        MATRIX_RESULT=generated,
                        PIPELINE_RESULT=pipeline,
                        PLATFORMS=platforms,
                    ),
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, expected, result.stderr)

    def run_common_gate(self, job, values):
        command = self.common[job]["steps"][0]["run"]
        for expression, value in values.items():
            command = command.replace("${{ " + expression + " }}", value)
        self.assertNotIn("${{", command)
        return subprocess.run(["bash", "-e", "-c", command], capture_output=True).returncode

    def test_enabled_test_summaries_reject_skips_and_cancellations(self):
        for kind in ("unit", "integration"):
            for status in STATUSES:
                with self.subTest(kind=kind, status=status):
                    code = self.run_common_gate(
                        f"{kind}_tests_complete", {f"needs.{kind}_tests.result": status}
                    )
                    self.assertEqual(code == 0, status == "success")

    def test_platform_gate_only_accepts_explicitly_disabled_skips(self):
        for build, unit, integration in itertools.product((False, True), repeat=3):
            values = {
                "needs.checkout_and_config.result": "success",
                "needs.checkout_and_config.outputs.test_artifact_enabled": str(build).lower(),
                "inputs.run_unit_tests": str(unit).lower(),
                "inputs.run_integration_tests": str(integration).lower(),
                "needs.build_test_artifact.result": "success" if build else "skipped",
                "needs.unit_tests_complete.result": "success" if unit else "skipped",
                "needs.integration_tests_complete.result": "success" if integration else "skipped",
            }
            self.assertEqual(self.run_common_gate("all_tests_complete", values), 0)
            for key in (
                "needs.checkout_and_config.result",
                "needs.build_test_artifact.result",
                "needs.unit_tests_complete.result",
                "needs.integration_tests_complete.result",
            ):
                bad_values = dict(values, **{key: "cancelled"})
                self.assertNotEqual(self.run_common_gate("all_tests_complete", bad_values), 0)
                if values[key] == "success":
                    bad_values[key] = "skipped"
                    self.assertNotEqual(self.run_common_gate("all_tests_complete", bad_values), 0)


if __name__ == "__main__":
    unittest.main()
