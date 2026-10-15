"""Hardware-independent checks for report delivery and performance failures."""

import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import upload_report
import generate_matrix
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "performance_smoke", ROOT / "tests/test_utils/run_performance_smoke.py"
)
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class ReportingTests(unittest.TestCase):
    def test_real_multipart_upload_without_user_id(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(
                    (self.headers, self.rfile.read(int(self.headers["Content-Length"])))
                )
                self.send_response(201)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                report = Path(directory) / "coverage.json"
                report.write_text('{"totals": {"percent_covered": 80}}')
                event = Path(directory) / "event.json"
                event.write_text(
                    json.dumps({"issue": {"number": 42, "pull_request": {"url": "x"}}})
                )
                with patch.dict(
                    os.environ,
                    {
                        "FLAGCICD_API_TOKEN": "test-token",
                        "GITHUB_REPOSITORY": "org/te",
                        "GITHUB_EVENT_PATH": str(event),
                    },
                ):
                    result = upload_report.upload(
                        report,
                        "coverage",
                        "coverage-metax-c500",
                        f"http://127.0.0.1:{server.server_port}/metrics/",
                    )
                self.assertIn("HTTP 201", result)
                headers, body = requests[0]
                self.assertEqual(headers["Authorization"], "Bearer test-token")
                self.assertIsNone(headers.get("x-user-id"))
                self.assertNotIn(b"user_id", body)
                for field, value in (
                    ("file_type", "coverage"),
                    ("pr_id", "42"),
                    ("git_project_name", "org/te"),
                ):
                    self.assertIn(f'name="{field}"\r\n\r\n{value}\r\n'.encode(), body)
                self.assertIn(report.read_bytes(), body)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_missing_inputs_do_not_send_requests(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            upload_report.subprocess, "run"
        ) as run:
            report = Path(directory) / "report.json"
            self.assertIn("missing", upload_report.upload(report, "coverage", "job", "unused"))
            report.write_text("{}")
            with patch.dict(os.environ, {"FLAGCICD_API_TOKEN": ""}):
                self.assertIn(
                    "unavailable", upload_report.upload(report, "coverage", "job", "unused")
                )
            run.assert_not_called()

    def test_http_transport_failures_and_timeout_remain_nonblocking(self):
        for result in (
            Mock(returncode=0, stdout="401"),
            Mock(returncode=7, stdout="000"),
            subprocess.TimeoutExpired("curl", 65),
        ):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as directory:
                report = Path(directory) / "report.json"
                report.write_text("{}")
                summary = Path(directory) / "summary"
                kwargs = (
                    {"side_effect": result}
                    if isinstance(result, Exception)
                    else {"return_value": result}
                )
                with patch.dict(
                    os.environ,
                    {
                        "FLAGCICD_API_TOKEN": "test-token",
                        "GITHUB_EVENT_PATH": "",
                        "GITHUB_STEP_SUMMARY": str(summary),
                    },
                ), patch.object(upload_report.subprocess, "run", **kwargs), patch(
                    "sys.argv",
                    [
                        "upload_report",
                        "--report",
                        str(report),
                        "--file-type",
                        "coverage",
                    ],
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ) as stdout:
                    self.assertEqual(upload_report.main(), 0)
                self.assertIn("FlagCICD report: Upload failed", summary.read_text())
                self.assertNotIn("test-token", stdout.getvalue())


class PerformanceSmokeTests(unittest.TestCase):
    def test_ascend_patch_runs_before_loading_accelerator_modules(self):
        apply_patch = Mock(side_effect=RuntimeError("patch invoked"))
        with patch.object(
            smoke.runpy, "run_path", return_value={"apply_ascend_npu_patch": apply_patch}
        ) as load_patch:
            with self.assertRaisesRegex(RuntimeError, "patch invoked"):
                smoke.run_suite({"environment": {"platform": "ascend"}}, 0, 1)
        load_patch.assert_called_once_with(str(ROOT / "tests/plugin/backend/npu/npu_patch.py"))
        apply_patch.assert_called_once_with()

    def test_all_discovered_platforms_enable_reports(self):
        configs = ROOT / ".github/configs"
        for platform in generate_matrix._load_watch_rules(configs):
            with self.subTest(platform=platform):
                config = yaml.safe_load((configs / f"{platform}.yml").read_text())
                self.assertTrue(config["coverage"]["enabled"])
                self.assertTrue(config["performance_smoke"]["enabled"])
                self.assertRegex(
                    config["performance_smoke"].get("file_type", "performance"),
                    r"^[a-zA-Z0-9_-]+$",
                )
                self.assertTrue(config["device_types"])

    def test_device_resolution_handles_vendor_aliases_and_rejects_cpu(self):
        for actual in ("cuda", "npu", "musa", "gcu", "xpu", "cpu"):
            with self.subTest(actual=actual):
                engine, torch = Mock(), Mock()
                engine.te_device_type.return_value = "cuda"
                engine.te_platform.return_value.is_available.return_value = True
                torch.empty.return_value.device.type = actual
                if actual == "cpu":
                    with self.assertRaises(RuntimeError):
                        smoke.resolve_device(engine, torch)
                else:
                    platform, device = smoke.resolve_device(engine, torch)
                    self.assertEqual(device, actual)
                    self.assertEqual(engine.TE_DEVICE_TYPE, actual)
                    self.assertIs(platform, engine.te_platform.return_value)
                torch.empty.assert_called_once_with(0, device="cuda")
        engine.te_platform.return_value.is_available.return_value = False
        torch.reset_mock()
        with self.assertRaises(RuntimeError):
            smoke.resolve_device(engine, torch)
        torch.empty.assert_not_called()

    def test_timing_synchronization_and_invalid_measurements(self):
        operation, synchronize = Mock(), Mock()
        with patch.object(smoke.time, "perf_counter", side_effect=[0, 0.001, 0.01, 0.013]):
            latency = smoke.measure(operation, synchronize, 2, 2)
        self.assertAlmostEqual(latency["median"], 2)
        self.assertEqual(operation.call_count, 4)
        self.assertEqual(synchronize.call_count, 3)
        for end in (0, -1, float("nan"), float("inf")):
            with patch.object(smoke.time, "perf_counter", side_effect=[0, end]):
                with self.assertRaises(RuntimeError):
                    smoke.measure(Mock(), Mock(), 0, 1)

    def test_suite_failure_and_empty_suite_write_failed_reports(self):
        for side_effect in (RuntimeError("accelerator unavailable"), None):
            with self.subTest(side_effect=side_effect), tempfile.TemporaryDirectory() as directory:
                report = Path(directory) / "nested/report.json"
                with patch.object(smoke, "run_suite", side_effect=side_effect), patch(
                    "sys.argv",
                    ["smoke", "--report", str(report), "--platform", "metax", "--device", "c500"],
                ), patch.dict(
                    os.environ, {"TE_TEST_COMMIT": "tested-sha"}
                ), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    self.assertEqual(smoke.main(), 1)
                data = json.loads(report.read_text())
                self.assertEqual(data["summary"]["status"], "failed")
                self.assertEqual(data["environment"]["commit"], "tested-sha")

    def test_complete_suite_succeeds_and_writes_summary(self):
        def suite(report, *args):
            report["cases"] = [
                {"name": f"case{i}", "status": "passed"} for i in range(smoke.EXPECTED_CASES)
            ]

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            summary = Path(directory) / "summary"
            with patch.object(smoke, "run_suite", side_effect=suite), patch(
                "sys.argv",
                ["smoke", "--report", str(report), "--platform", "metax", "--device", "c500"],
            ), patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}):
                self.assertEqual(smoke.main(), 0)
            self.assertEqual(
                json.loads(report.read_text())["summary"]["passed"], smoke.EXPECTED_CASES
            )
            self.assertIn(
                f"{smoke.EXPECTED_CASES}/{smoke.EXPECTED_CASES} passed", summary.read_text()
            )


if __name__ == "__main__":
    unittest.main()
