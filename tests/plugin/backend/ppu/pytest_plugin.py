"""PPU-only, collection-time skips; unexpected runtime failures stay failures."""
import fnmatch
import json
import os
from pathlib import Path

import pytest


def _matches(rule, item):
    if not fnmatch.fnmatchcase(item.nodeid, rule["pattern"]):
        return False
    params = getattr(getattr(item, "callspec", None), "params", {})
    for key, values in rule.get("params", {}).items():
        value = params.get(key)
        if not any(value == candidate or str(value) == candidate for candidate in values):
            return False
    for key, lengths in rule.get("param_length", {}).items():
        if len(params.get(key, ())) not in lengths:
            return False
    if rule.get("missing_checkpoint"):
        artifact_dir = Path(os.environ.get(
            "NVTE_TEST_CHECKPOINT_ARTIFACT_PATH", "artifacts/tests/pytorch/test_checkpoint"
        ))
        if (artifact_dir / (params["name"] + ".pt")).is_file():
            return False
    if "min_devices" in rule:
        import torch
        if torch.cuda.device_count() < rule["min_devices"]:
            return False
    return True


def pytest_collection_modifyitems(items):
    rules = json.loads(Path(__file__).with_name("config.json").read_text())["skips"]
    for item in items:
        for rule in rules:
            if _matches(rule, item):
                item.add_marker(pytest.mark.skip(reason=f"{rule['id']}: {rule['reason']}"))
                break


def pytest_configure(config):
    if os.environ.get("PLATFORM") != "ppu":
        raise pytest.UsageError("Use the PPU container environment from .github/configs/ppu.yml")
    # Match per-step environment in the shared QA scripts, before test imports.
    if any(str(arg).endswith("test_cpu_offloading_v1.py") for arg in config.args):
        os.environ["NVTE_CPU_OFFLOAD_V1"] = "1"
    if any(str(arg).endswith("test_onnx_export.py") for arg in config.args):
        os.environ["NVTE_UnfusedDPA_Emulate_FP8"] = "1"
