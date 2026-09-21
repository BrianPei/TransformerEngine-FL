"""Run PPU QA serially, preserving failures and per-file JUnit reports."""
import hashlib
from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
SUITES = json.loads(CONFIG_PATH.read_text())["suites"]
ALIASES = {
    "pytorch_debug": "debug",
    "pytorch_unittest": "unittest",
    "pytorch_distributed_unittest": "distributed",
    "pytorch_onnx_unittest": "onnx",
}


def main():
    groups = [ALIASES.get(x, x) for x in sys.argv[1:]] or list(SUITES)
    if any(x not in SUITES for x in groups):
        raise SystemExit(f"Usage: {sys.argv[0]} [{'|'.join(SUITES)} ...]")
    root = Path(os.environ["XML_LOG_DIR"])
    config = yaml.safe_load((HERE.parents[3] / ".github/configs/ppu.yml").read_text())
    nproc = config["nproc_per_node"]
    if type(nproc) is not int or nproc < 2:
        raise ValueError("ppu.yml: nproc_per_node must be an integer >= 2")
    distributed_env = os.environ.copy()
    if "distributed" in groups:
        import torch

        devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = devices.split(",") if devices else [str(i) for i in range(torch.cuda.device_count())]
        if len(devices) < nproc or torch.cuda.device_count() < nproc:
            raise ValueError("Not enough visible PPUs for nproc_per_node")
        # Shared tests derive their launch size from the visible device count.
        distributed_env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:nproc])
    failed = False
    results = []
    for group in groups:
        dest = root / group
        dest.mkdir(parents=True, exist_ok=True)
        for target in SUITES[group]:
            env = distributed_env if group == "distributed" else os.environ.copy()
            name = target.replace("/", "_").removesuffix(".py")
            xml = dest / (name + ".xml")
            cmd = [
                sys.executable, "-m", "pytest", target, "-v", "--tb=short", "-ra",
                "-o", "faulthandler_timeout=120", f"--junitxml={xml}",
                "-p", "tests.plugin.backend.ppu.pytest_plugin",
            ]
            if group == "debug":
                cmd += [
                    "--feature_dirs=transformer_engine/debug/features",
                    "--configs_dir=tests/pytorch/debug/test_configs/",
                ]
            if os.environ.get("PPU_PROBE") == "1":
                cmd += ["--maxfail=2"]
            print(f"[RUN] {group}: {target}", flush=True)
            xml.unlink(missing_ok=True)
            started = time.monotonic()
            timed_out = False
            config_hash = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
            timeout = int(os.environ.get("PPU_STEP_TIMEOUT", "7200"))
            log_path = dest / (name + ".log")
            with log_path.open("w") as log:
                proc = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env
                )
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    print(f"TIMEOUT after {timeout} seconds", file=log)
                    suite = ET.Element("testsuite", name=target, tests="1", errors="1")
                    case = ET.SubElement(suite, "testcase", name="runner_timeout")
                    ET.SubElement(case, "error", message=f"File exceeded {timeout}s")
                    ET.ElementTree(suite).write(xml, encoding="unicode")
            print(f"[EXIT {proc.returncode}] {target}", flush=True)
            counts = {}
            if xml.exists():
                counts = {
                    key: sum(int(e.get(key, 0)) for e in ET.parse(xml).iter("testsuite"))
                    for key in ("tests", "failures", "errors", "skipped")
                }
            # Pytest can return 5 for a module-level capability skip. Accept it
            # only when JUnit proves a skip and contains no collection errors.
            valid_exit = proc.returncode == 0 or (
                proc.returncode == 5 and counts.get("skipped", 0) > 0
                and counts.get("errors", 0) == 0 and counts.get("failures", 0) == 0
            )
            ok = (
                counts.get("tests", 0) > 0 and not timed_out and valid_exit
                and counts.get("failures", 0) == 0 and counts.get("errors", 0) == 0
            )
            if not ok:
                print(f"[FAIL] {target}: {log_path}", flush=True)
                with log_path.open(errors="replace") as log:
                    print("".join(deque(log, maxlen=80)), flush=True)
            failed |= not ok
            results.append(dict(
                suite=group, target=target, returncode=proc.returncode, timeout=timed_out,
                ok=ok, counts=counts, seconds=round(time.monotonic() - started, 2),
                command=cmd, config_sha256=config_hash,
                cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES", "all"),
                nproc_per_node=nproc if group == "distributed" else None,
            ))
            (root / ("summary-" + "-".join(groups) + ".json")).write_text(
                json.dumps(results, indent=2) + "\n"
            )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
