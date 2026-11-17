#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/set_env.sh"
# Individual pytest failures are collected and propagated after all targets.
set +e
cd "$TE_PATH"
PYTHON="${PYTHON_BIN:-python3}"
ROOT_LOG_DIR="$XML_LOG_DIR"
mkdir -p "$ROOT_LOG_DIR"
SUMMARY="$ROOT_LOG_DIR/summary.tsv"
printf 'suite\ttarget\treturncode\tstatus\treason\towner\trestore_condition\n' > "$SUMMARY"
OVERALL_FAIL=0
run_target() {
  local suite="$1" target="$2"; shift 2
  local name="${target//\//_}"; name="${name%.py}"
  local log="$XML_LOG_DIR/${name}.log" xml="$XML_LOG_DIR/${name}.xml"
  local timeout_seconds="${PPU_STEP_TIMEOUT:-7200}"
  if ! [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "PPU_STEP_TIMEOUT must be a positive integer, got: $timeout_seconds" >&2
    OVERALL_FAIL=1
    printf '%s\t%s\t2\tfailed\t\t\t\n' "$suite" "$target" >> "$SUMMARY"
    return
  fi
  local -a cmd=(
    env TE_TEST_GROUP_PRESET=plugin-unit
    "$PYTHON" "$TE_PATH/tests/test_utils/run_ci_test_group.py"
    --target-env "$target"
    "$PYTHON" -m pytest "$TE_PATH/$target" -v -s --tb=short -ra -o faulthandler_timeout=120 --junitxml="$xml"
  )
  cmd+=("$@")
  echo "[RUN][$suite] $target"
  rm -f "$xml"
  local rc status
  timeout --signal=TERM --kill-after=10s "${timeout_seconds}s" "${cmd[@]}" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  status=passed
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    status=timed_out
    OVERALL_FAIL=1
    "$PYTHON" - "$xml" "$target" "$timeout_seconds" <<'PY'
import pathlib, sys, xml.etree.ElementTree as ET

xml = pathlib.Path(sys.argv[1])
target = sys.argv[2]
timeout_seconds = sys.argv[3]
suite = ET.Element('testsuite', name=target, tests='1', failures='0', errors='1', skipped='0')
case = ET.SubElement(suite, 'testcase', name='runner_timeout')
ET.SubElement(case, 'error', message=f'Target exceeded {timeout_seconds} seconds')
ET.ElementTree(suite).write(xml, encoding='unicode', xml_declaration=True)
PY
    if [ "$?" -ne 0 ]; then
      echo "Failed to write timeout JUnit report: $xml" >&2
    fi
  elif [ "$rc" -ne 0 ]; then
    status=failed
    OVERALL_FAIL=1
  fi
  local reason="" line
  if [ "$rc" -eq 0 ]; then
    while IFS= read -r line; do
      case "$line" in
        *"No test files in this invocation match the FlagCICD plan"* | \
        *"No selected FlagCICD test nodes remain"*)
          status=ignored
          reason="Not selected or ignored by the FlagCICD plan"
          break
          ;;
      esac
    done < "$log"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t\t\n' "$suite" "$target" "$rc" "$status" "$reason" >> "$SUMMARY"
  echo "[EXIT $rc][$suite] $target (log: $log)"
}
# debug, distributed and onnx run the shared qa scripts like every other
# platform; PPU differences live only in .github/configs/ppu.yml ignored_tests.
shared_qa_script() {
  case "$1" in
    debug) echo qa/L0_pytorch_debug_unittest/test.sh ;;
    distributed) echo qa/L1_pytorch_distributed_unittest/test.sh ;;
    onnx) echo qa/L1_pytorch_onnx_unittest/test.sh ;;
  esac
}
run_shared_suite() {
  local suite="$1" script rc status=passed
  script="$(shared_qa_script "$suite")"
  echo "[RUN][$suite] $script"
  timeout --signal=TERM --kill-after=10s "${PPU_STEP_TIMEOUT:-7200}s" \
    env XML_LOG_DIR="$XML_LOG_DIR" bash "$TE_PATH/$script" 2>&1 | tee "$XML_LOG_DIR/shared-suite.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then status=timed_out; OVERALL_FAIL=1
  elif [ "$rc" -ne 0 ]; then status=failed; OVERALL_FAIL=1; fi
  printf '%s\t%s\t%s\t%s\t\t\t\n' "$suite" "$script" "$rc" "$status" >> "$SUMMARY"
  echo "[EXIT $rc][$suite] $script"
}
run_unittest_suite() {
  # Per-test quarantines are maintained in .github/configs/ppu.yml.
  run_target unittest tests/pytorch/test_sanity.py
  run_target unittest tests/pytorch/test_recipe.py
  run_target unittest tests/pytorch/test_deferred_init.py
  run_target unittest tests/pytorch/test_numerics.py
  run_target unittest tests/pytorch/test_cuda_graphs.py
  run_target unittest tests/pytorch/test_jit.py
  run_target unittest tests/pytorch/nvfp4
  run_target unittest tests/pytorch/test_quantized_tensor.py
  run_target unittest tests/pytorch/test_float8blockwisetensor.py
  run_target unittest tests/pytorch/test_float8_blockwise_scaling_exact.py
  run_target unittest tests/pytorch/test_float8_blockwise_gemm_exact.py
  run_target unittest tests/pytorch/test_gqa.py
  # Enable the full upstream PyTorch RoPE file, including standard RoPE,
  # THD, fused QKV and autocast checks, in addition to the plugin smoke tests.
  # Validated on PPU with transformer-engine-fl:v2.17.0-thead-ci after
  # `compiler triton`; the fused operators selected the default.flagos backend.
  # Unfiltered run: 9609 collected, 5865 passed, 3744 skipped, 0 failures
  # in 7244.54 seconds (2:00:44). The skips are test-internal parameter guards:
  # - margin=0 with start_positions=True (no valid random offset range);
  # - QKV cp_size>1 with start_positions=True (unsupported combination);
  # - QKV seq_length<margin (would produce a negative input length).
  # These guards apply across platforms; they are not PPU CI ignores.
  # No RoPE failure quarantine is needed in .github/configs/ppu.yml for this run.
  run_target unittest tests/pytorch/test_fused_rope.py
  run_target unittest tests/pytorch/test_fused_optimizer.py
  run_target unittest tests/pytorch/test_multi_tensor.py
  run_target unittest tests/pytorch/test_fusible_ops.py
  run_target unittest tests/pytorch/test_permutation.py
  run_target unittest tests/pytorch/test_parallel_cross_entropy.py
  run_target unittest tests/pytorch/test_cpu_offloading.py
  export NVTE_CPU_OFFLOAD_V1=1
  run_target unittest tests/pytorch/test_cpu_offloading_v1.py
  run_target unittest tests/pytorch/attention/test_attention.py
  run_target unittest tests/pytorch/attention/test_kv_cache.py
  run_target unittest tests/pytorch/test_hf_integration.py
  local plugin_target plugin_targets
  plugin_targets=$(TE_TEST_GROUP_PRESET=plugin-unit "$PYTHON" "$TE_PATH/tests/test_utils/run_ci_test_group.py" --list-targets) || { OVERALL_FAIL=1; return; }
  while IFS= read -r plugin_target; do
    [ -n "$plugin_target" ] && run_target unittest "$plugin_target"
  done <<< "$plugin_targets"
}

run_suite() {
  local suite="$1"
  local previous_visible="${CUDA_VISIBLE_DEVICES:-}"
  local previous_visible_set=0
  [ -n "${CUDA_VISIBLE_DEVICES+x}" ] && previous_visible_set=1
  if [ "$suite" = distributed ]; then
    local nproc
    nproc="$($PYTHON -c 'import yaml; print(yaml.safe_load(open(".github/configs/ppu.yml"))["nproc_per_node"])')"
    if ! [[ "$nproc" =~ ^[0-9]+$ ]] || [ "$nproc" -lt 2 ]; then
      echo "ppu.yml: nproc_per_node must be an integer >= 2" >&2
      OVERALL_FAIL=1
      printf '%s\t<device-check>\t2\tfailed\t\t\t\n' "$suite" >> "$SUMMARY"
      return
    fi
    local hardware_count
    hardware_count="$($PYTHON -c 'import torch; print(torch.cuda.device_count())')"
    local visible="${CUDA_VISIBLE_DEVICES:-}"
    local -a devices=()
    if [ -n "$visible" ]; then
      IFS=',' read -r -a devices <<< "$visible"
    else
      local index
      for ((index=0; index<hardware_count; index++)); do devices+=("$index"); done
    fi
    if [ "$hardware_count" -lt "$nproc" ] || [ "${#devices[@]}" -lt "$nproc" ]; then
      echo "PPU distributed requires $nproc visible devices, found ${#devices[@]} (runtime count: $hardware_count)" >&2
      OVERALL_FAIL=1
      printf '%s\t<device-check>\t1\tfailed\t\t\t\n' "$suite" >> "$SUMMARY"
      return
    fi
    local selected="${devices[0]}"
    local index
    for ((index=1; index<nproc; index++)); do selected+=",${devices[$index]}"; done
    export CUDA_VISIBLE_DEVICES="$selected"
    echo "[distributed] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (nproc_per_node=$nproc)"
  fi
  mkdir -p "$ROOT_LOG_DIR/$suite"
  local old_log="$XML_LOG_DIR"; XML_LOG_DIR="$ROOT_LOG_DIR/$suite"
  if [ -n "$(shared_qa_script "$suite")" ]; then
    run_shared_suite "$suite"
    XML_LOG_DIR="$old_log"
    if [ "$suite" = distributed ]; then
      if [ "$previous_visible_set" -eq 1 ]; then export CUDA_VISIBLE_DEVICES="$previous_visible"
      else unset CUDA_VISIBLE_DEVICES; fi
    fi
    return
  fi
  run_unittest_suite
  XML_LOG_DIR="$old_log"
  if [ "$suite" = distributed ]; then
    if [ "$previous_visible_set" -eq 1 ]; then
      export CUDA_VISIBLE_DEVICES="$previous_visible"
    else
      unset CUDA_VISIBLE_DEVICES
    fi
  fi
}
if [ "$#" -eq 0 ]; then set -- debug unittest distributed onnx; fi
for suite in "$@"; do
  case "$suite" in
    debug|unittest|distributed|onnx) run_suite "$suite" ;;
    -h|--help) echo "Usage: $0 [debug] [unittest] [distributed] [onnx]"; exit 0 ;;
    *) echo "Unknown suite: $suite" >&2; exit 2 ;;
  esac
done
summary_rc=0
"$PYTHON" - "$SUMMARY" "$ROOT_LOG_DIR/summary.json" <<'PY' || summary_rc=$?
import json, os, pathlib, re, sys, tempfile, xml.etree.ElementTree as ET

summary = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
rows = []
with summary.open() as f:
    next(f)
    for line in f:
        suite, target, rc, status, reason, owner, restore = line.rstrip('\n').split('\t')
        xml = summary.parent / suite / (target.replace('/', '_').removesuffix('.py') + '.xml')
        log = summary.parent / suite / (target.replace('/', '_').removesuffix('.py') + '.log')
        counts = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
        if target.startswith('qa/'):
            # A shared qa script writes one JUnit file per step into the suite dir.
            log = summary.parent / suite / 'shared-suite.log'
            for report in sorted((summary.parent / suite).glob('*.xml')):
                root = ET.parse(report)
                for key in counts: counts[key] += sum(int(x.get(key, 0)) for x in root.iter('testsuite'))
        elif status != 'ignored':
            if not xml.is_file():
                raise FileNotFoundError(f'Missing JUnit report: {xml}')
            root = ET.parse(xml)
            for key in counts: counts[key] = sum(int(x.get(key, 0)) for x in root.iter('testsuite'))
            if status == 'passed' and counts['tests'] == 0:
                raise ValueError(f'Passing target collected zero tests: {target}')
        counts['deselected'] = 0
        if log.exists():
            matches = re.findall(r'(\d+) deselected', log.read_text(errors='replace'))
            if matches: counts['deselected'] = int(matches[-1])
        counts.update(
            suite=suite,
            target=target,
            returncode=int(rc) if rc else None,
            status=status,
            reason=reason or None,
            owner=owner or None,
            restore_condition=restore or None,
        )
        rows.append(counts)
with tempfile.NamedTemporaryFile('w', dir=destination.parent, delete=False) as f:
    json.dump(rows, f, indent=2)
    f.write('\n')
    temporary = f.name
os.replace(temporary, destination)
PY
if [ "$summary_rc" -ne 0 ]; then
  echo "Failed to generate test summary from JUnit reports" >&2
  OVERALL_FAIL=1
fi
exit "$OVERALL_FAIL"
