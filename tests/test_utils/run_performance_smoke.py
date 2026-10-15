#!/usr/bin/env python3
"""Small BF16 TE benchmark suite; always write a report on ordinary failures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import time
import traceback


OPERATORS = ("Linear", "LayerNorm", "RMSNorm")
MODES = ("forward", "forward_backward")
EXPECTED_CASES = len(OPERATORS) * len(MODES)


def measure(operation, synchronize, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        operation()
    synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        operation()
        synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Timing must be finite and positive")
        samples.append(elapsed)
    return {
        "unit": "ms",
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def resolve_device(engine, torch) -> tuple:
    platform = engine.te_platform()
    if not platform.is_available():
        raise RuntimeError("An accelerator is required; CPU fallback is disabled")
    requested = engine.te_device_type()
    # Some vendor runtimes accept 'cuda' but return a GCU/XPU tensor. TE's
    # public Linear validates the tensor type against this process-local global.
    actual = torch.empty(0, device=requested).device.type
    if actual == "cpu":
        raise RuntimeError("An accelerator is required; CPU fallback is disabled")
    engine.TE_DEVICE_TYPE = actual
    return platform, actual


def run_suite(report: dict, warmup: int, iterations: int) -> None:
    if report["environment"]["platform"] == "ascend":
        patch_path = Path(__file__).resolve().parents[1] / "plugin/backend/npu/npu_patch.py"
        patch = runpy.run_path(str(patch_path))
        patch["apply_ascend_npu_patch"]()

    import torch
    import transformer_engine as engine
    import transformer_engine.pytorch as te

    platform, device = resolve_device(engine, torch)
    torch.manual_seed(42)
    report["environment"].update(
        {
            "torch_version": torch.__version__,
            "te_version": getattr(engine, "__version__", "unknown"),
            "device_type": device,
            "device_name": platform.get_device_name(0),
        }
    )
    for operator in OPERATORS:
        for mode in MODES:
            case = {
                "name": f"{operator}_{mode}",
                "operator": operator,
                "mode": mode,
                "shape": [256, 512],
                "dtype": "bfloat16",
                "status": "failed",
            }
            report["cases"].append(case)
            try:
                kwargs = {"device": device, "params_dtype": torch.bfloat16}
                layer = (
                    te.Linear(512, 512, **kwargs)
                    if operator == "Linear"
                    else getattr(te, operator)(512, **kwargs)
                )
                backward = mode == "forward_backward"
                x = torch.randn(
                    256, 512, device=device, dtype=torch.bfloat16, requires_grad=backward
                )
                gradient = torch.ones_like(x)

                def operation():
                    if backward:
                        layer.zero_grad(set_to_none=True)
                        x.grad = None
                        y = layer(x)
                        y.backward(gradient)
                    else:
                        with torch.no_grad():
                            y = layer(x)
                    return y

                y = operation()
                platform.synchronize()
                if not torch.isfinite(y).all().item():
                    raise RuntimeError("Nonfinite output")
                if backward:
                    gradients = [x.grad, *(p.grad for p in layer.parameters() if p.requires_grad)]
                    if any(g is None or not torch.isfinite(g).all().item() for g in gradients):
                        raise RuntimeError("Missing or nonfinite gradients")
                case["latency"] = measure(operation, platform.synchronize, warmup, iterations)
                case["status"] = "passed"
                print(f"{case['name']}: {case['latency']['median']:.4f} ms", flush=True)
            except Exception as error:
                case["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    report = {
        "schema_version": 1,
        "report_type": "performance_smoke",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": args.platform,
            "device": args.device,
            "commit": os.environ.get("TE_TEST_COMMIT", os.environ.get("GITHUB_SHA", "")),
            "run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
            "image": os.environ.get("TE_CI_IMAGE", ""),
        },
        "measurement": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "method": "synchronized_wall_clock",
            "includes_host_overhead": True,
        },
        "cases": [],
    }
    try:
        run_suite(report, args.warmup, args.iterations)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    passed = sum(case["status"] == "passed" for case in report["cases"])
    success = len(report["cases"]) == passed == EXPECTED_CASES and "error" not in report
    report["summary"] = {
        "total": len(report["cases"]),
        "passed": passed,
        "failed": len(report["cases"]) - passed,
        "status": "passed" if success else "failed",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(
                f"Performance smoke: {passed}/{EXPECTED_CASES} passed"
                f" ({args.platform}/{args.device})\n\n"
            )
            stream.write("| Case | Status | Median latency (ms) |\n| --- | --- | --- |\n")
            for case in report["cases"]:
                latency = case.get("latency", {}).get("median")
                value = f"{latency:.4f}" if latency is not None else "—"
                stream.write(f"| {case['name']} | {case['status']} | {value} |\n")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
