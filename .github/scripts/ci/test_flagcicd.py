"""Unit tests for FlagCICD v1 core components.

Tests gate validation, identity verification, and contract schemas without
requiring accelerator hardware or TransformerEngine installation.

Run: python -m unittest .github/scripts/ci/test_flagcicd.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

# Import modules under test
import flagcicd_gate as gate
import flagcicd_identity as identity
import flagcicd_contract as contract
import flagcicd_planner as planner


ROOT = Path(__file__).resolve().parents[3]
SHA = "a" * 40


def make_plan(platforms=(), test_groups=(), blocked=(), stage="pr"):
    """Return a hash-valid plan with the fields the gate subcommands read."""
    plan = {
        "schema_version": contract.PLAN_SCHEMA,
        "repository": "TransformerEngine-FL",
        "stage": stage,
        "selection": "affected",
        "lint_only": False,
        "source_sha": SHA,
        "base_sha": None,
        "test_level": "L2",
        "risk_level": "R1",
        "risk_rule_version": "v2",
        "changed_files": {"count": 0, "documentation_only": False},
        "common_path_triggers": [],
        "platforms": list(platforms),
        "test_groups": list(test_groups),
        "excluded": [],
        "blocked": list(blocked),
        "reasons": [],
        "coverage_policy": {"method": "absolute"},
    }
    plan["plan_hash"] = contract.compute_plan_hash(plan)
    return plan


def make_group(name, lane, platform="cuda", device="a100", evidence="junit", kind="unit"):
    return {
        "platform": platform,
        "device": device,
        "group": name,
        "kind": kind,
        "level": "L0" if lane == "preflight" else "L1",
        "required": True,
        "lane": lane,
        "evidence": evidence,
        "resources": {
            "class": "single_device",
            "devices": 0 if lane == "preflight" else (8 if lane == "serial" else 1),
            "exclusive": lane == "serial",
            "timeout_minutes": 90 if lane == "preflight" else 30,
        },
        "coverage": lane == "parallel",
        "exec": (None if lane == "preflight" else {"path": f"run_{name}.sh", "runner": "script"}),
    }


def platform_report(plan, platform, **overrides):
    report = {
        "schema_version": contract.PLATFORM_REPORT_SCHEMA,
        "repository": plan["repository"],
        "stage": plan["stage"],
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "platform": platform,
        "status": "passed",
        "errors": [],
    }
    report.update(overrides)
    return report


class ContractSchemaTests(unittest.TestCase):
    """Test contract schema constants and validation."""

    def test_schema_version_constants(self):
        """Verify schema version constants are defined."""
        self.assertEqual(contract.PLAN_SCHEMA, "flagcicd.plan/v1")
        self.assertEqual(contract.GROUP_REPORT_SCHEMA, "flagcicd.group-report/v1")
        self.assertEqual(contract.PLATFORM_REPORT_SCHEMA, "flagcicd.platform-report/v1")
        self.assertEqual(contract.GATE_SCHEMA, "flagcicd.gate/v1")
        self.assertEqual(contract.IDENTITY_SCHEMA, "flagcicd.artifact-identity/v1")

    def test_status_constants(self):
        """Verify status constants are defined."""
        self.assertIn("passed", contract.STATUSES)
        self.assertIn("test_failed", contract.STATUSES)
        self.assertIn("build_failed", contract.STATUSES)
        self.assertIn("infra_error", contract.STATUSES)

    def test_level_and_risk_constants(self):
        """Verify level and risk constants."""
        self.assertEqual(contract.LEVELS, ("L0", "L1", "L2", "L3", "L4"))
        self.assertEqual(contract.RISKS, ("R1", "R2", "R3", "R4"))

    def test_canonical_json_deterministic(self):
        """Verify canonical JSON is deterministic."""
        obj = {"z": 1, "a": 2, "m": [3, 2, 1]}
        json1 = contract.canonical_json(obj)
        json2 = contract.canonical_json(obj)
        self.assertEqual(json1, json2)
        self.assertIn('"a":2', json1)
        self.assertTrue(json1.index('"a"') < json1.index('"z"'))

    def test_compute_plan_hash(self):
        """Verify plan hash computation."""
        plan = {
            "schema_version": "flagcicd.plan/v1",
            "source_sha": "a" * 40,
            "stage": "pr",
            "platforms": ["cuda"],
            "test_groups": [],
        }
        hash1 = contract.compute_plan_hash(plan)
        self.assertEqual(len(hash1), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in hash1))

        # Same plan should produce same hash
        hash2 = contract.compute_plan_hash(plan)
        self.assertEqual(hash1, hash2)

        # Different plan should produce different hash
        plan["stage"] = "nightly"
        hash3 = contract.compute_plan_hash(plan)
        self.assertNotEqual(hash1, hash3)


class IdentityTests(unittest.TestCase):
    """Test artifact identity verification."""

    def test_write_identity_creates_valid_json(self):
        """Test writing identity file with required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            (artifact_dir / "test.whl").write_bytes(b"wheel")

            # Use real argparse Namespace with all required fields
            args = argparse.Namespace(
                artifact_dir=str(artifact_dir),
                platform="cuda",
                source_sha="a" * 40,
                config_hash="b" * 64,
                coverage_active=True,
                test_marker="unittest",
                image_ref="flagscale/te-cuda:latest",
                image_digest="sha256:abc123",
                test_level="L1",
                timestamp="2024-01-01T00:00:00Z",
            )

            identity.cmd_write(args)

            identity_file = artifact_dir / "artifact-identity.json"
            self.assertTrue(identity_file.exists())
            data = json.loads(identity_file.read_text())

            self.assertEqual(data["schema_version"], contract.IDENTITY_SCHEMA)
            self.assertEqual(data["platform"], "cuda")
            self.assertEqual(data["source_sha"], "a" * 40)
            self.assertEqual(data["config_hash"], "b" * 64)
            self.assertTrue(data["coverage_instrumentation"])
            self.assertEqual(data["image_ref"], "flagscale/te-cuda:latest")
            self.assertEqual(identity.cmd_verify(args), 0)

    def test_verify_identity_accepts_matching_identity(self):
        """Test verification passes with matching identity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            identity_file = artifact_dir / "artifact-identity.json"
            (artifact_dir / "test.whl").write_bytes(b"wheel")
            identity_file.write_text(
                json.dumps(
                    {
                        "schema_version": contract.IDENTITY_SCHEMA,
                        "platform": "cuda",
                        "source_sha": "a" * 40,
                        "config_hash": "b" * 64,
                        "coverage_active": False,
                        "test_marker": None,
                        "wheels": [
                            {
                                "name": "test.whl",
                                "size": 5,
                                "sha256": hashlib.sha256(b"wheel").hexdigest(),
                            }
                        ],
                        "toolchain": {},
                    }
                )
            )

            args = Mock(
                artifact_dir=str(artifact_dir),
                platform="cuda",
                source_sha="a" * 40,
                config_hash="b" * 64,
            )

            # Should not raise
            result = identity.cmd_verify(args)
            self.assertEqual(result, 0)

    def test_verify_identity_rejects_platform_mismatch(self):
        """Test verification fails on platform mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            identity_file = artifact_dir / "artifact-identity.json"
            (artifact_dir / "test.whl").write_bytes(b"wheel")
            identity_file.write_text(
                json.dumps(
                    {
                        "schema_version": contract.IDENTITY_SCHEMA,
                        "platform": "cuda",
                        "source_sha": "a" * 40,
                        "config_hash": "b" * 64,
                        "coverage_active": False,
                        "test_marker": None,
                        "wheels": [
                            {
                                "name": "test.whl",
                                "size": 5,
                                "sha256": hashlib.sha256(b"wheel").hexdigest(),
                            }
                        ],
                        "toolchain": {},
                    }
                )
            )

            args = Mock(
                artifact_dir=str(artifact_dir),
                platform="ascend",  # Mismatch
                source_sha="a" * 40,
                config_hash="b" * 64,
            )

            result = identity.cmd_verify(args)
            self.assertNotEqual(result, 0)

    def test_verify_identity_rejects_sha_mismatch(self):
        """Test verification fails on source_sha mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            identity_file = artifact_dir / "artifact-identity.json"
            (artifact_dir / "test.whl").write_bytes(b"wheel")
            identity_file.write_text(
                json.dumps(
                    {
                        "schema_version": contract.IDENTITY_SCHEMA,
                        "platform": "cuda",
                        "source_sha": "a" * 40,
                        "config_hash": "b" * 64,
                        "coverage_active": False,
                        "test_marker": None,
                        "wheels": [
                            {
                                "name": "test.whl",
                                "size": 5,
                                "sha256": hashlib.sha256(b"wheel").hexdigest(),
                            }
                        ],
                        "toolchain": {},
                    }
                )
            )

            args = Mock(
                artifact_dir=str(artifact_dir),
                platform="cuda",
                source_sha="c" * 40,  # Mismatch
                config_hash="b" * 64,
            )

            result = identity.cmd_verify(args)
            self.assertNotEqual(result, 0)

    def test_payload_integrity(self):
        for change in (
            "unchanged",
            "same_size",
            "truncated",
            "missing",
            "extra",
            "duplicate",
            "escape",
            "empty",
            "schema",
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                wheel = folder / "test.whl"
                wheel.write_bytes(b"original")
                manifest = {
                    "schema_version": contract.IDENTITY_SCHEMA,
                    "platform": "cuda",
                    "source_sha": "a" * 40,
                    "config_hash": "b" * 64,
                    "wheels": [
                        {
                            "name": wheel.name,
                            "size": 8,
                            "sha256": hashlib.sha256(b"original").hexdigest(),
                        }
                    ],
                }
                if change == "same_size":
                    wheel.write_bytes(b"replaced")
                if change == "truncated":
                    wheel.write_bytes(b"x")
                if change == "missing":
                    wheel.unlink()
                if change == "extra":
                    (folder / "old.whl").write_bytes(b"old")
                if change == "duplicate":
                    manifest["wheels"] *= 2
                if change == "escape":
                    manifest["wheels"][0]["name"] = "../test.whl"
                if change == "empty":
                    manifest["wheels"] = []
                if change == "schema":
                    manifest["schema_version"] = "unknown"
                (folder / "artifact-identity.json").write_text(json.dumps(manifest))
                args = argparse.Namespace(
                    artifact_dir=folder,
                    platform="cuda",
                    source_sha="a" * 40,
                    config_hash="b" * 64,
                )
                self.assertEqual(identity.cmd_verify(args), 0 if change == "unchanged" else 1)


class GateSliceTests(unittest.TestCase):
    """cmd_slice partitions one platform's plan into lane matrices."""

    def run_slice(self, plan, config_text="device_types: [a100]\n"):
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = Path(tmpdir, "configs")
            configs.mkdir()
            (configs / "cuda.yml").write_text(config_text)
            output = Path(tmpdir, "github-output")
            output.write_text("")
            args = argparse.Namespace(platform="cuda", configs_dir=str(configs), github_output=True)
            head = Mock(stdout=SHA + "\n")
            env = {"FLAGCICD_PLAN": json.dumps(plan), "GITHUB_OUTPUT": str(output)}
            with patch.object(gate.subprocess, "run", return_value=head), patch.dict(
                gate.os.environ, env
            ):
                code = gate.cmd_slice(args)
            values = dict(
                line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
            )
        return code, values

    def plan_for(self, groups, config_text="device_types: [a100]\n"):
        entry = {
            "name": "cuda",
            "config_hash": hashlib.sha256(config_text.encode()).hexdigest(),
            "coverage_required": True,
        }
        other = make_group("pytorch_unittest", "parallel", platform="ascend", device="910b")
        return make_plan(platforms=[entry], test_groups=[*groups, other])

    def test_lanes_are_partitioned_per_platform(self):
        plan = self.plan_for(
            [
                make_group("preflight", "preflight"),
                make_group("pytorch_unittest", "parallel"),
                make_group("pytorch_mcore_integration", "serial", evidence="exit_code"),
            ]
        )
        code, out = self.run_slice(plan)
        self.assertEqual(code, 0)
        names = {
            lane: [entry["name"] for entry in json.loads(out[f"{lane}_matrix"])["include"]]
            for lane in ("preflight", "parallel", "serial")
        }
        self.assertEqual(
            names,
            {
                "preflight": ["preflight"],
                "parallel": ["pytorch_unittest"],
                "serial": ["pytorch_mcore_integration"],
            },
        )
        self.assertEqual(
            (out["has_parallel"], out["has_serial"], out["preflight_selected"]),
            ("true", "true", "true"),
        )
        self.assertEqual((out["coverage_active"], out["coverage_required"]), ("true", "true"))
        preflight = json.loads(out["preflight_matrix"])["include"][0]
        self.assertEqual((preflight["device"], preflight["timeout"]), ("a100", 90))
        serial = json.loads(out["serial_matrix"])["include"][0]
        self.assertEqual(
            (serial["path"], serial["devices"]), ("run_pytorch_mcore_integration.sh", 8)
        )

    def test_empty_lanes_report_false(self):
        code, out = self.run_slice(self.plan_for([make_group("preflight", "preflight")]))
        self.assertEqual(code, 0)
        self.assertEqual((out["has_parallel"], out["has_serial"]), ("false", "false"))
        self.assertEqual(json.loads(out["parallel_matrix"]), {"include": []})

    def test_unknown_lane_fails(self):
        code, _ = self.run_slice(self.plan_for([make_group("probe", "fast")]))
        self.assertEqual(code, 1)

    def test_config_hash_mismatch_fails(self):
        plan = self.plan_for([make_group("preflight", "preflight")])
        code, _ = self.run_slice(plan, config_text="device_types: [h100]\n")
        self.assertEqual(code, 1)

    def test_pytest_config_survives_planner_slice_and_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            probe = folder / "pytest_probe.py"
            probe.write_text(
                """import os, sys
from pathlib import Path
assert "--group-option" in sys.argv and "--step-option" in sys.argv
assert os.environ["GROUP_ENV"] == "group" and os.environ["STEP_ENV"] == "step"
xml = next(arg.split("=",1)[1] for arg in sys.argv if arg.startswith("--junitxml="))
Path(xml).write_text('<testsuite tests="1"><testcase name="probe"/></testsuite>')
"""
            )
            project = planner.load_project(ROOT / ".flagcicd/project.yaml")
            platforms = planner.load_platforms(ROOT / ".github/configs")
            platforms["cuda"]["config"]["unit_test_matrix"] = [
                {
                    "name": "pytorch_unittest",
                    "runner": "pytest",
                    "max_retries": 0,
                    "pytest_args": ["--group-option"],
                    "log_dir": str(folder / "logs"),
                    "env": {"GROUP_ENV": "group"},
                    "steps": [
                        {
                            "name": "probe",
                            "targets": ["probe_target"],
                            "args": ["--step-option"],
                            "env": {"STEP_ENV": "step"},
                            "junit": "probe.xml",
                        }
                    ],
                    # Execution metadata must not override the catalog's required flag.
                    "required": False,
                }
            ]
            groups = planner.build_test_groups(project, platforms, ["cuda"], "L1", [])
            code, outputs = self.run_slice(self.plan_for(groups))
            self.assertEqual(code, 0)
            entry = json.loads(outputs["serial_matrix"])["include"][0]
            self.assertTrue(entry["required"])
            env = dict(
                os.environ,
                TE_TEST_GROUP_JSON=json.dumps(entry),
                TE_TEST_PYTEST_COMMAND=f"{sys.executable} {probe}",
            )
            result = subprocess.run(
                [sys.executable, str(ROOT / "tests/test_utils/run_ci_test_group.py")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(gate.parse_junit_status(str(folder / "logs")), ("passed", None))
            group = next(g for g in groups if g["group"] == "pytorch_unittest")
            plan = make_plan(test_groups=[group])
            args = argparse.Namespace(
                platform="cuda",
                device="a100",
                group="pytorch_unittest",
                kind="unit",
                level="L1",
                required=True,
                exit_code_file=None,
                junit_dir=str(folder / "wrong"),
                duration="0",
                output=folder / "group-report.json",
            )
            steps = {
                "checkout": {"outcome": "success"},
                "execute": {"outcome": "success"},
            }
            with patch.dict(
                os.environ,
                FLAGCICD_PLAN=json.dumps(plan),
                FLAGCICD_STEPS=json.dumps(steps),
            ):
                self.assertEqual(gate.cmd_group_report(args), 0)
            self.assertEqual(json.loads(args.output.read_text())["status"], "passed")


class GroupReportEvidenceTests(unittest.TestCase):
    """A group's declared evidence decides how a zero exit code is judged."""

    def test_missing_junit_is_not_reported_as_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, failure_stage = gate.parse_junit_status(str(Path(tmpdir) / "missing"))
        self.assertEqual((status, failure_stage), ("report_error", "junit_missing"))

    def test_empty_junit_directory_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, failure_stage = gate.parse_junit_status(tmpdir)
        self.assertEqual((status, failure_stage), ("skipped", "no_tests_collected"))

    def report(self, group, evidence, outcome="success", exit_code="0", plan_group=None):
        plan = make_plan(test_groups=[plan_group or make_group(group, "serial", evidence=evidence)])
        steps = {"checkout": {"outcome": "success"}, "execute": {"outcome": outcome}}
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_file = Path(tmpdir, "exit_code")
            exit_file.write_text(exit_code)
            junit = Path(tmpdir, "junit")
            junit.mkdir()
            output = Path(tmpdir, "group-report.json")
            args = argparse.Namespace(
                platform="cuda",
                device="a100",
                group=group,
                kind="integration",
                level="L2",
                required=True,
                exit_code_file=exit_file,
                junit_dir=str(junit),
                duration="5",
                output=output,
            )
            env = {
                "FLAGCICD_PLAN": json.dumps(plan),
                "FLAGCICD_STEPS": json.dumps(steps),
            }
            with patch.dict(gate.os.environ, env):
                self.assertEqual(gate.cmd_group_report(args), 0)
            return json.loads(output.read_text())

    def test_exit_code_group_passes_without_junit(self):
        report = self.report("pytorch_mcore_integration", "exit_code")
        self.assertEqual((report["status"], report["executed"]), ("passed", True))

    def test_junit_group_without_tests_is_skipped(self):
        report = self.report("pytorch_unittest", "junit")
        self.assertEqual(
            (report["status"], report["failure_stage"]),
            ("skipped", "no_tests_collected"),
        )

    def test_nonzero_exit_fails_regardless_of_evidence(self):
        report = self.report("pytorch_mcore_integration", "exit_code", "failure", "1")
        self.assertEqual(
            (report["status"], report["failure_stage"]),
            ("test_failed", "test_execution"),
        )

    def test_group_missing_from_plan_is_report_error(self):
        other = make_group("pytorch_unittest", "parallel")
        report = self.report("unplanned", "exit_code", plan_group=other)
        self.assertEqual((report["status"], report["failure_stage"]), ("report_error", "plan"))

    def test_unknown_evidence_is_report_error(self):
        report = self.report("pytorch_unittest", "stdout")
        self.assertEqual((report["status"], report["failure_stage"]), ("report_error", "evidence"))

    def parse(self, xml):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "report.xml").write_text(xml)
            return gate.parse_junit_status(tmp)

    def test_malformed_reports_cannot_pass(self):
        reports = [
            '<testsuite tests="1" failures="0"><testcase><failure/></testcase></testsuite>',
            '<testsuite tests="1"/>',
            '<testsuite tests="-1"/>',
            '<testsuite tests="oops"/>',
            '<testsuite tests="2"><testcase/></testsuite>',
            "<testsuite><testcase><skipped/><error/></testcase></testsuite>",
            "<unknown><testcase/></unknown>",
            "<testsuite>",
        ]
        for xml in reports:
            with self.subTest(xml=xml):
                self.assertEqual(self.parse(xml), ("report_error", "junit_parsing"))

    def test_actual_case_outcomes_without_summary_attributes(self):
        for tag in ("failure", "error"):
            with self.subTest(tag=tag):
                self.assertEqual(
                    self.parse(f"<testsuite><testcase><{tag}/></testcase></testsuite>"),
                    ("test_failed", "test_execution"),
                )
        self.assertEqual(
            self.parse("<testsuite><testcase><skipped/></testcase></testsuite>"),
            ("skipped", "all_skipped"),
        )

    def test_nested_namespaced_suites_are_counted_once(self):
        xml = """<testsuites xmlns="urn:junit" tests="2" skipped="1">
          <testsuite tests="2" skipped="1"><testsuite tests="1"><testcase/></testsuite>
          <testcase><skipped/></testcase></testsuite></testsuites>"""
        self.assertEqual(self.parse(xml), ("passed", None))


class GatePlatformReportTests(unittest.TestCase):
    """Test gate platform-report validation logic."""

    def test_required_coverage_cannot_be_skipped(self):
        plan = {
            "schema_version": contract.PLAN_SCHEMA,
            "repository": "TransformerEngine-FL",
            "stage": "pr",
            "selection": "affected",
            "lint_only": False,
            "source_sha": "a" * 40,
            "base_sha": None,
            "test_level": "L1",
            "risk_level": "R1",
            "risk_rule_version": "v1",
            "changed_files": {"count": 0, "documentation_only": False},
            "common_path_triggers": [],
            "platforms": [
                {
                    "name": "ascend",
                    "coverage_required": True,
                    "device_types": ["ascend"],
                }
            ],
            "test_groups": [],
            "excluded": [],
            "blocked": [],
            "reasons": [],
            "coverage_policy": {"method": "absolute"},
        }
        plan["plan_hash"] = contract.compute_plan_hash(plan)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "platform-report.json"
            args = argparse.Namespace(
                platform="ascend",
                reports_dir=Path(tmpdir),
                coverage_result="skipped",
                coverage_required=True,
                output=output,
            )
            with patch.dict(
                gate.os.environ,
                {"FLAGCICD_PLAN": contract.canonical_json(plan)},
                clear=False,
            ):
                self.assertEqual(gate.cmd_platform(args), 1)
            report = json.loads(output.read_text())
            self.assertTrue(any("coverage" in error.lower() for error in report["errors"]))

    def test_optional_failures_and_missing_reports_allow_final_gate(self):
        for required, include_report in (
            (False, True),
            (False, False),
            (True, True),
            (True, False),
        ):
            with self.subTest(
                required=required, include_report=include_report
            ), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                group = make_group("probe", "parallel")
                group["required"] = required
                plan = make_plan(platforms=[{"name": "cuda"}], test_groups=[group])
                report = {
                    "schema_version": contract.GROUP_REPORT_SCHEMA,
                    "platform": "cuda",
                    "device": "a100",
                    "group": "probe",
                    "status": "test_failed",
                    "executed": True,
                    "plan_hash": plan["plan_hash"],
                    "source_sha": plan["source_sha"],
                }
                if include_report:
                    (folder / "group-report.json").write_text(json.dumps(report))
                args = argparse.Namespace(
                    platform="cuda",
                    reports_dir=folder,
                    coverage_required=False,
                    coverage_result="failure",
                    output=folder / "platform-report.json",
                )
                with patch.dict(os.environ, FLAGCICD_PLAN=json.dumps(plan)):
                    code = gate.cmd_platform(args)
                self.assertEqual(code, int(required))
                platform_report = json.loads(args.output.read_text())
                code, _ = GateTests().run_gate(plan, [platform_report])
                self.assertEqual(code, int(required))

    def test_coverage_collection_failure_only_blocks_when_required(self):
        for required in (False, True):
            with self.subTest(required=required), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                plan = make_plan(platforms=[{"name": "cuda"}])
                args = argparse.Namespace(
                    platform="cuda",
                    reports_dir=folder,
                    coverage_required=required,
                    coverage_result="failure",
                    output=folder / "platform-report.json",
                )
                with patch.dict(os.environ, FLAGCICD_PLAN=json.dumps(plan)):
                    self.assertEqual(gate.cmd_platform(args), int(required))

    def test_workflow_failure_policy_is_on_concrete_jobs(self):
        tests = yaml.safe_load((ROOT / ".github/workflows/test_groups_common.yml").read_text())
        coverage = yaml.safe_load((ROOT / ".github/workflows/all_tests_common.yml").read_text())
        self.assertEqual(
            tests["jobs"]["test_group"]["continue-on-error"],
            "${{ matrix.required != true }}",
        )
        self.assertEqual(
            coverage["jobs"]["coverage"]["continue-on-error"],
            "${{ needs.checkout_and_config.outputs.coverage_required != 'true' }}",
        )


class GateTests(unittest.TestCase):
    """cmd_gate fails closed on every report/plan inconsistency."""

    def run_gate(self, plan, reports, pipeline="success"):
        with tempfile.TemporaryDirectory() as tmpdir:
            reports_dir = Path(tmpdir, "reports")
            for index, report in enumerate(reports):
                path = reports_dir / f"flagcicd-platform-{index}" / "platform-report.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(report))
            output = Path(tmpdir, "gate.json")
            args = argparse.Namespace(
                plan_job="plan",
                pipeline_job="platform-pipeline",
                reports_dir=reports_dir,
                output=output,
            )
            needs = {
                "plan": {"result": "success"},
                "platform-pipeline": {"result": pipeline},
            }
            env = {
                "FLAGCICD_PLAN": json.dumps(plan),
                "FLAGCICD_NEEDS": json.dumps(needs),
            }
            with patch.dict(gate.os.environ, env):
                code = gate.cmd_gate(args)
            return code, json.loads(output.read_text())

    def setUp(self):
        self.plan = make_plan(platforms=[{"name": "cuda"}, {"name": "ascend"}])

    def test_all_reports_pass(self):
        reports = [
            platform_report(self.plan, "cuda"),
            platform_report(self.plan, "ascend"),
        ]
        code, summary = self.run_gate(self.plan, reports)
        self.assertEqual((code, summary["status"], summary["errors"]), (0, "passed", []))

    def test_missing_platform_report_fails(self):
        code, summary = self.run_gate(self.plan, [platform_report(self.plan, "cuda")])
        self.assertEqual(code, 1)
        self.assertIn("Missing platform report: ascend", summary["errors"])

    def test_duplicate_platform_report_fails(self):
        reports = [
            platform_report(self.plan, "cuda"),
            platform_report(self.plan, "cuda"),
            platform_report(self.plan, "ascend"),
        ]
        code, summary = self.run_gate(self.plan, reports)
        self.assertEqual(code, 1)
        self.assertIn("Duplicate platform reports: cuda", summary["errors"])

    def test_plan_hash_mismatch_fails(self):
        reports = [
            platform_report(self.plan, "cuda", plan_hash="0" * 64),
            platform_report(self.plan, "ascend"),
        ]
        code, summary = self.run_gate(self.plan, reports)
        self.assertEqual(code, 1)
        self.assertIn("Platform cuda: plan_hash mismatch", summary["errors"])

    def test_required_blocked_group_fails(self):
        blocked = [{"platform": "cuda", "group": "model_validation", "required": True}]
        plan = make_plan(platforms=[{"name": "cuda"}], blocked=blocked, stage="release")
        code, summary = self.run_gate(plan, [platform_report(plan, "cuda")])
        self.assertEqual(code, 1)
        self.assertTrue(any("Required groups are blocked" in e for e in summary["errors"]))

    def test_optional_blocked_group_does_not_fail(self):
        blocked = [{"platform": "cuda", "group": "pytorch_benchmark", "required": False}]
        plan = make_plan(platforms=[{"name": "cuda"}], blocked=blocked)
        code, _ = self.run_gate(plan, [platform_report(plan, "cuda")])
        self.assertEqual(code, 0)

    def test_failed_pipeline_job_fails(self):
        reports = [
            platform_report(self.plan, "cuda"),
            platform_report(self.plan, "ascend"),
        ]
        code, summary = self.run_gate(self.plan, reports, pipeline="failure")
        self.assertEqual(code, 1)
        self.assertIn("Pipeline job must be success", summary["errors"])


class RegressionTests(unittest.TestCase):
    """flagcicd_regression accepts the raw coverage json the workflow writes."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "flagcicd_regression", ROOT / ".github/scripts/flagcicd_regression.py"
        )
        cls.regression = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.regression)

    def run_cli(self, current, *extra):
        with tempfile.TemporaryDirectory() as tmpdir:
            current_path = Path(tmpdir, "current.json")
            current_path.write_text(json.dumps({"totals": {"percent_covered": current}}))
            args = []
            for flag, value in zip(extra[::2], extra[1::2]):
                path = Path(tmpdir, flag.strip("-") + ".json")
                path.write_text(json.dumps(value))
                args += [flag, str(path)]
            output = Path(tmpdir, "verdict.json")
            result = subprocess.run(
                [
                    "python3",
                    str(ROOT / ".github/scripts/flagcicd_regression.py"),
                    "--current",
                    str(current_path),
                    *args,
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            return result.returncode, json.loads(output.read_text())["result"]

    def test_raw_coverage_json_against_raw_baseline(self):
        code, result = self.run_cli(70.0, "--baseline", {"totals": {"percent_covered": 75.0}})
        self.assertEqual((code, result["status"]), (1, "regression"))
        code, result = self.run_cli(74.0, "--baseline", {"totals": {"percent_covered": 75.0}})
        self.assertEqual((code, result["status"]), (0, "acceptable"))

    def test_zero_variance_history_flags_any_drop(self):
        code, result = self.run_cli(79.9, "--historical", [80.0] * 8)
        self.assertEqual((code, result["status"]), (1, "outlier_below"))
        self.assertIn("zero variance", result["message"])
        code, result = self.run_cli(80.0, "--historical", [80.0] * 8)
        self.assertEqual((code, result["status"]), (0, "within_bounds"))

    def test_unknown_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir, "report.json")
            path.write_text(json.dumps({"schema_version": "other/v1"}))
            with self.assertRaises(ValueError):
                self.regression.load_report(path)


if __name__ == "__main__":
    unittest.main()
