#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/set_env.sh"
source "$SCRIPT_DIR/config.sh"
# Individual pytest failures are collected and propagated after all targets.
set +e
cd "$TE_PATH"
PYTHON="${PYTHON_BIN:-python3}"
ROOT_LOG_DIR="$XML_LOG_DIR"
mkdir -p "$ROOT_LOG_DIR"
SUMMARY="$ROOT_LOG_DIR/summary.tsv"
printf 'suite\ttarget\treturncode\tstatus\n' > "$SUMMARY"
OVERALL_FAIL=0
run_target() {
  local suite="$1" target="$2"; shift 2
  local name="${target//\//_}"; name="${name%.py}"
  local log="$XML_LOG_DIR/${name}.log" xml="$XML_LOG_DIR/${name}.xml"
  local -a cmd=("$PYTHON" -m pytest "$TE_PATH/$target" -v -s --tb=short -ra -o faulthandler_timeout="${PPU_STEP_TIMEOUT:-7200}" --junitxml="$xml")
  cmd+=("$@")
  echo "[RUN][$suite] $target"
  rm -f "$xml"
  local rc status
  "${cmd[@]}" > >(tee "$log") 2>&1
  rc=${PIPESTATUS[0]}
  status=passed
  if [ "$rc" -ne 0 ]; then status=failed; OVERALL_FAIL=1; fi
  printf '%s\t%s\t%s\t%s\n' "$suite" "$target" "$rc" "$status" >> "$SUMMARY"
  echo "[EXIT $rc][$suite] $target (log: $log)"
}
run_suite() {
  local suite="$1" target
  local -n targets="PPU_${suite^^}_TARGETS"
  if [ "$suite" = distributed ]; then
    local nproc="$(python3 -c 'import yaml; print(yaml.safe_load(open(".github/configs/ppu.yml"))["nproc_per_node"])')"
    local visible="${CUDA_VISIBLE_DEVICES:-}"
    if [ -n "$visible" ]; then
      local count=${visible//[^,]/}; count=$(( ${#count} + 1 ))
    else
      count="$(python3 -c 'import torch; print(torch.cuda.device_count())')"
    fi
    if [ "$count" -lt "$nproc" ]; then
      echo "PPU distributed requires $nproc visible devices, found $count" >&2
      OVERALL_FAIL=1
      printf '%s\t<device-check>\t1\tfailed\n' "$suite" >> "$SUMMARY"
      return
    fi
  fi
  mkdir -p "$ROOT_LOG_DIR/$suite"
  local old_log="$XML_LOG_DIR"; XML_LOG_DIR="$ROOT_LOG_DIR/$suite"
  for target in "${targets[@]}"; do
    local -a extra=()
    [ "$suite" = debug ] && extra+=(--feature_dirs=transformer_engine/debug/features --configs_dir=tests/pytorch/debug/test_configs/)
    [ "$target" = tests/pytorch/test_cpu_offloading_v1.py ] && export NVTE_CPU_OFFLOAD_V1=1
    [ "$target" = tests/pytorch/test_onnx_export.py ] && export NVTE_UnfusedDPA_Emulate_FP8=1
    [ -n "${PPU_SKIP_K[$target]:-}" ] && extra+=(-k "not (${PPU_SKIP_K[$target]})")
    run_target "$suite" "$target" "${extra[@]}"
  done
  XML_LOG_DIR="$old_log"
}
if [ "$#" -eq 0 ]; then set -- debug unittest distributed onnx; fi
for suite in "$@"; do
  case "$suite" in
    debug|unittest|distributed|onnx) run_suite "$suite" ;;
    -h|--help) echo "Usage: $0 [debug] [unittest] [distributed] [onnx]"; exit 0 ;;
    *) echo "Unknown suite: $suite" >&2; exit 2 ;;
  esac
done
"$PYTHON" - "$SUMMARY" "$ROOT_LOG_DIR/summary.json" <<'PY'
import json, sys
rows=[]
with open(sys.argv[1]) as f:
    next(f)
    for line in f:
        suite,target,rc,status=line.rstrip('\n').split('\t')
        import pathlib, re, xml.etree.ElementTree as ET
        xml = pathlib.Path(sys.argv[1]).parent / suite / (target.replace('/', '_').removesuffix('.py') + '.xml')
        log = pathlib.Path(sys.argv[1]).parent / suite / (target.replace('/', '_').removesuffix('.py') + '.log')
        counts = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
        if xml.exists():
            root = ET.parse(xml)
            for key in counts: counts[key] = sum(int(x.get(key, 0)) for x in root.iter('testsuite'))
        counts['deselected'] = 0
        if log.exists():
            matches = re.findall(r'(\d+) deselected', log.read_text(errors='replace'))
            if matches: counts['deselected'] = int(matches[-1])
        counts.update(suite=suite, target=target, returncode=int(rc), status=status)
        rows.append(counts)
with open(sys.argv[2], 'w') as f: json.dump(rows, f, indent=2); f.write('\n')
PY
exit "$OVERALL_FAIL"
