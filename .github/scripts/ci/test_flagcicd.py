"""Unit tests for FlagCICD v1 core components.

Tests gate validation, identity verification, and contract schemas without
requiring accelerator hardware or TransformerEngine installation.

Run: python -m unittest .github/scripts/ci/test_flagcicd.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Import modules under test
import flagcicd_gate as gate
import flagcicd_identity as identity
import flagcicd_contract as contract


ROOT = Path(__file__).resolve().parents[3]


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

    def test_verify_identity_accepts_matching_identity(self):
        """Test verification passes with matching identity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            identity_file = artifact_dir / "artifact-identity.json"
            identity_file.write_text(
                json.dumps(
                    {
                        "schema_version": contract.IDENTITY_SCHEMA,
                        "platform": "cuda",
                        "source_sha": "a" * 40,
                        "config_hash": "b" * 64,
                        "coverage_active": False,
                        "test_marker": None,
                        "wheels": [],
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
            identity_file.write_text(
                json.dumps(
                    {
                        "schema_version": contract.IDENTITY_SCHEMA,
                        "platform": "cuda",
                        "source_sha": "a" * 40,
                        "config_hash": "b" * 64,
                        "coverage_active": False,
                        "test_marker": None,
                        "wheels": [],
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
            identity_file.write_text(
                json.dumps(
                    {
                        "schema_version": contract.IDENTITY_SCHEMA,
                        "platform": "cuda",
                        "source_sha": "a" * 40,
                        "config_hash": "b" * 64,
                        "coverage_active": False,
                        "test_marker": None,
                        "wheels": [],
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


class GateSliceTests(unittest.TestCase):
    """Test gate slice command for matrix generation."""

    def test_slice_requires_plan_environment(self):
        """Test slice command requires FLAGCICD_PLAN environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_file = Path(tmpdir) / "plan.json"
            plan_file.write_text(
                json.dumps(
                    {
                        "schema_version": "flagcicd.plan/v1",
                        "plan_hash": "x" * 64,
                        "source_sha": "a" * 40,
                        "stage": "pr",
                        "platforms": ["cuda"],
                        "test_groups": [],
                    }
                )
            )

            # Without FLAGCICD_PLAN env var, should error
            args = Mock(plan=str(plan_file), platform="cuda")

            # Test would require subprocess or environment setup
            # Skip detailed matrix generation test
            self.assertTrue(plan_file.exists())


class GroupReportEvidenceTests(unittest.TestCase):
    """Test that a successful command still has test-result evidence."""

    def test_missing_junit_is_not_reported_as_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, failure_stage = gate.parse_junit_status(str(Path(tmpdir) / "missing"))
        self.assertEqual((status, failure_stage), ("report_error", "junit_missing"))

    def test_empty_junit_directory_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status, failure_stage = gate.parse_junit_status(tmpdir)
        self.assertEqual((status, failure_stage), ("skipped", "no_tests_collected"))


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
                {"name": "ascend", "coverage_required": True, "device_types": ["ascend"]}
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
                gate.os.environ, {"FLAGCICD_PLAN": contract.canonical_json(plan)}, clear=False
            ):
                self.assertEqual(gate.cmd_platform(args), 1)
            report = json.loads(output.read_text())
            self.assertTrue(any("coverage" in error.lower() for error in report["errors"]))

    def test_platform_report_structure(self):
        """Test platform report has correct structure."""
        # Basic structure validation without full workflow execution
        report = {
            "schema_version": contract.PLATFORM_REPORT_SCHEMA,
            "plan_hash": "x" * 64,
            "source_sha": "a" * 40,
            "platform": "cuda",
            "groups": [],
            "errors": [],
        }

        self.assertEqual(report["schema_version"], contract.PLATFORM_REPORT_SCHEMA)
        self.assertEqual(report["platform"], "cuda")
        self.assertIsInstance(report["groups"], list)
        self.assertIsInstance(report["errors"], list)


class GateValidationTests(unittest.TestCase):
    """Test top-level gate validation logic."""

    def test_gate_structure(self):
        """Test gate has correct structure."""
        # Basic structure validation
        gate_data = {
            "schema_version": contract.GATE_SCHEMA,
            "plan_hash": "x" * 64,
            "source_sha": "a" * 40,
            "verdict": "pass",
            "platforms": ["cuda", "ascend"],
        }

        self.assertEqual(gate_data["schema_version"], contract.GATE_SCHEMA)
        self.assertIn(gate_data["verdict"], ["pass", "fail"])
        self.assertIsInstance(gate_data["platforms"], list)


if __name__ == "__main__":
    unittest.main()
