"""Run PPU QA serially, preserving failures and per-file JUnit reports."""

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
SUITES = {
    "debug": [
        "tests/pytorch/debug/test_sanity.py",
        "tests/pytorch/debug/test_config.py",
        "tests/pytorch/debug/test_numerics.py",
        "tests/pytorch/debug/test_log.py",
        "tests/pytorch/debug/test_api_features.py",
        "tests/pytorch/debug/test_perf.py",
    ],
    "unittest": [
        "tests/pytorch/test_sanity.py",
        "tests/pytorch/test_recipe.py",
        "tests/pytorch/test_deferred_init.py",
        "tests/pytorch/test_numerics.py",
        "tests/pytorch/test_cuda_graphs.py",
        "tests/pytorch/test_jit.py",
        "tests/pytorch/test_fused_rope.py",
        "tests/pytorch/nvfp4",
        "tests/pytorch/test_quantized_tensor.py",
        "tests/pytorch/test_float8blockwisetensor.py",
        "tests/pytorch/test_float8_blockwise_scaling_exact.py",
        "tests/pytorch/test_float8_blockwise_gemm_exact.py",
        "tests/pytorch/test_gqa.py",
        "tests/pytorch/test_fused_optimizer.py",
        "tests/pytorch/test_multi_tensor.py",
        "tests/pytorch/test_fusible_ops.py",
        "tests/pytorch/test_permutation.py",
        "tests/pytorch/test_parallel_cross_entropy.py",
        "tests/pytorch/test_cpu_offloading.py",
        "tests/pytorch/test_cpu_offloading_v1.py",
        "tests/pytorch/attention/test_attention.py",
        "tests/pytorch/attention/test_kv_cache.py",
        "tests/pytorch/test_hf_integration.py",
        "tests/pytorch/test_checkpoint.py",
        "tests/plugin/plugin/test_policy.py",
        "tests/plugin/plugin/test_manager.py",
        "tests/plugin/plugin/test_policy_selection.py",
        "tests/plugin/backend/flagos/test_lifecycle.py",
        "tests/plugin/backend/flagos/test_fused_rope.py",
        "tests/plugin/backend/flagos/test_optimizer.py",
        "tests/plugin/backend/flagos/test_gemm.py",
        "tests/plugin/backend/flagos/test_multi_tensor.py",
        "tests/plugin/backend/flagos/test_rmsnorm.py",
        "tests/plugin/backend/flagos/test_softmax.py",
        "tests/plugin/backend/reference/test_lifecycle.py",
        "tests/plugin/backend/reference/test_activation.py",
        "tests/plugin/backend/reference/test_dropout.py",
        "tests/plugin/backend/reference/test_gemm.py",
    ],
    "distributed": [
        "tests/pytorch/distributed/test_numerics.py",
        "tests/pytorch/distributed/test_numerics_exact.py",
        "tests/pytorch/distributed/test_torch_fsdp2.py",
        "tests/pytorch/distributed/test_cast_master_weights_to_fp8.py",
        "tests/pytorch/attention/test_cp_utils.py",
    ],
    "onnx": ["tests/pytorch/test_onnx_export.py"],
}
K_SKIP_EXPRESSIONS = {
    "tests/pytorch/debug/test_sanity.py": (
        "test_sanity_debug and fake_quant and False and (mha_attention or transformer_layer)"
    ),
    "tests/pytorch/debug/test_api_features.py": (
        "test_per_tensor_scaling or test_fake_quant or "
        "test_statistics_collection or test_statistics_multi_run"
    ),
    "tests/pytorch/test_sanity.py": "test_sanity_grouped_linear and (1-dtype or 2-dtype)",
    "tests/pytorch/test_numerics.py": (
        "(test_gpt_cuda_graph and (dtype1 or dtype2)) or "
        "(test_layernorm_accuracy and (dtype1 or dtype2)) or "
        "(test_transformer_layer_hidden_states_format and 126m-2-dtype)"
    ),
    "tests/pytorch/test_cuda_graphs.py": (
        "test_make_graphed_callables_with_kwargs or "
        "(test_make_graphed_callables and (transformer or mha) "
        "and (dtype1 or dtype2)) or "
        "(test_make_graphed_callables_with_dot_product_attention "
        "and (dtype1 or dtype2))"
    ),
    "tests/pytorch/test_fused_optimizer.py": "TestFusedSGD or test_bf16_exp_avg_and_exp_avg_sq",
    "tests/pytorch/test_multi_tensor.py": "test_multi_tensor_compute_scale_and_scale_inv",
    "tests/pytorch/test_fusible_ops.py": (
        "test_grouped_linear or test_backward_add_rmsnorm or test_grouped_mlp or "
        "(test_basic_linear and not test_basic_linear_quantized "
        "and (in_shape0 or in_shape1 or in_shape2)) or "
        "(test_layer_norm and not test_layer_norm_autocast and (dtype1 or dtype2)) or "
        "(test_rmsnorm and True and (dtype1 or dtype2)) or "
        "(test_activation and dtype1 and (qgelu or qgeglu or "
        "(glu and not (geglu or reglu or sreglu or swiglu)))) or "
        "(test_clamped_swiglu and dtype1) or "
        "(test_dropout and (dtype1 or dtype2) and shape2 and True and 0.5) or "
        "(test_forward_linear_bias_activation and (dtype1 or dtype2) "
        "and (in_shape0 or in_shape2)) or "
        "(test_forward_linear_bias_add and dtype1 and True) or "
        "(test_forward_linear_scale_add and dtype1 and (2.5 or 3.5)) or "
        "(TestCheckpointing and test_linear and True)"
    ),
    "tests/pytorch/test_permutation.py": "test_permutation",
    "tests/pytorch/test_cpu_offloading.py": (
        "(test_memory and (multihead_attention or transformer_layer)) or "
        "(test_numerics and transformer_layer) or "
        "(test_numerics and UnfusedAttention and True-multihead_attention)"
    ),
    "tests/pytorch/test_cpu_offloading_v1.py": (
        "test_cpu_offload and (multihead_attention or transformer_layer)"
    ),
    "tests/pytorch/attention/test_attention.py": "test_attention",
    "tests/pytorch/attention/test_kv_cache.py": "test_kv_cache",
    "tests/pytorch/test_checkpoint.py": "TestLoadCheckpoint and test_module",
}

ALIASES = {
    "pytorch_debug": "debug",
    "pytorch_unittest": "unittest",
    "pytorch_distributed_unittest": "distributed",
    "pytorch_onnx_unittest": "onnx",
}


def main():
    if os.environ.get("PLATFORM") != "ppu":
        raise RuntimeError("Use the PPU container environment from .github/configs/ppu.yml")
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
        devices = (
            devices.split(",") if devices else [str(i) for i in range(torch.cuda.device_count())]
        )
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
                sys.executable,
                "-m",
                "pytest",
                target,
                "-v",
                "--tb=short",
                "-ra",
                "-o",
                "faulthandler_timeout=120",
                f"--junitxml={xml}",
            ]
            if group == "debug":
                cmd += [
                    "--feature_dirs=transformer_engine/debug/features",
                    "--configs_dir=tests/pytorch/debug/test_configs/",
                ]
            skip_expression = K_SKIP_EXPRESSIONS.get(target)
            if skip_expression:
                cmd += ["-k", f"not ({skip_expression})"]
            if target.endswith("test_cpu_offloading_v1.py"):
                env["NVTE_CPU_OFFLOAD_V1"] = "1"
            if target.endswith("test_onnx_export.py"):
                env["NVTE_UnfusedDPA_Emulate_FP8"] = "1"
            if os.environ.get("PPU_PROBE") == "1":
                cmd += ["--maxfail=2"]
            print(f"[RUN] {group}: {target}", flush=True)
            xml.unlink(missing_ok=True)
            started = time.monotonic()
            timed_out = False
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
            intentionally_deselected = (
                bool(skip_expression)
                and proc.returncode == 5
                and counts.get("tests", 0) == 0
                and counts.get("errors", 0) == 0
            )
            valid_exit = (
                proc.returncode == 0
                or (
                    proc.returncode == 5
                    and counts.get("skipped", 0) > 0
                    and counts.get("errors", 0) == 0
                    and counts.get("failures", 0) == 0
                )
                or intentionally_deselected
            )
            ok = (
                (counts.get("tests", 0) > 0 or intentionally_deselected)
                and not timed_out
                and valid_exit
                and counts.get("failures", 0) == 0
                and counts.get("errors", 0) == 0
            )
            if not ok:
                print(f"[FAIL] {target}: {log_path}", flush=True)
                with log_path.open(errors="replace") as log:
                    print("".join(deque(log, maxlen=80)), flush=True)
            failed |= not ok
            results.append(
                dict(
                    suite=group,
                    target=target,
                    returncode=proc.returncode,
                    timeout=timed_out,
                    ok=ok,
                    counts=counts,
                    seconds=round(time.monotonic() - started, 2),
                    command=cmd,
                    cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES", "all"),
                    nproc_per_node=nproc if group == "distributed" else None,
                )
            )
            (root / ("summary-" + "-".join(groups) + ".json")).write_text(
                json.dumps(results, indent=2) + "\n"
            )
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
