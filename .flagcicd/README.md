# FlagCICD Integration for TransformerEngine-FL

## Overview

This directory contains the FlagCICD contract and configuration for TransformerEngine-FL.

- `project.yaml`: test level capabilities, risk mapping rules, and coverage contract
- `golden_baseline.json`: regression detection thresholds (absolute and statistical)

## Quick Start

### Trigger via GitHub UI

1. Go to **Actions** → **FlagCICD Entry** → **Run workflow**
2. Set parameters:
   - `platforms`: JSON array, e.g. `["cuda"]` or `["cuda", "ascend"]`
   - `test_level`: `L0` (preflight), `L1` (unit), `L2` (unit+integration), `L3` (full regression), `L4` (release)
   - `risk_level`: `auto` (uses risk mapper) or explicit `R1`-`R4`

### Test Levels

| Level | Purpose | Scope | Timeout |
|-------|---------|-------|---------|
| L0 | Preflight | Checkout + config validation | 15 min |
| L1 | Unit tests | Python unit tests | 60 min |
| L2 | Integration | Unit + integration tests | 360 min |
| L3 | Full regression | L2 + FA version compatibility + benchmarks | 480 min |
| L4 | Release validation | L3 + cross-repo integration + model validation | 2880 min |

**Note on L3**: Flash Attention version compatibility tests require Hopper or higher GPUs. These tests will be skipped if hardware is unavailable.

**Note on L4**: Requires cross-repository coordination with Megatron-LM-FL and vLLM-plugin-FL teams. Manual coordination required for release validation procedures.

### Trigger via repository_dispatch

```bash
gh api repos/$OWNER/$REPO/dispatches \
  -F event_type=flagcicd \
  -F 'client_payload[platforms]=["cuda"]' \
  -F 'client_payload[test_level]=L1'
```

## Test Levels

| Level | Purpose | Scope |
|-------|---------|-------|
| L0 | Preflight | Checkout and config validation |
| L1 | Unit tests | Python unit tests |
| L2 | Integration | Unit + integration tests |

## Risk Mapping Rules (v1)

Changed files are automatically mapped to minimum test levels:

| Rule | Paths | Min Level | Risk |
|------|-------|-----------|------|
| native-extension | `transformer_engine/**/csrc/**`, `**/cuda/**` | L3 | R3 |
| ci-and-build | `.github/**`, `CMakeLists.txt`, `setup.py` | L2 | R2 |
| numerical-and-performance | `tests/pytorch/**`, `qa/**` | L1 | R1 |

Unknown paths escalate to L2/R2.

## Regression Detection

Regression detection supports two modes and is automatically enabled for PR builds:

Baseline reports are fetched through the GitHub REST API by
`.github/scripts/fetch_flagcicd_baseline.py`. The check does not require the
vendor test image to contain the `gh` CLI. Callers must grant `actions: read`.

**Absolute threshold** (currently enabled):
```bash
python3 .github/scripts/flagcicd_regression.py \
  --current report.json \
  --baseline baseline.json \
  --threshold 2.0 \
  --output verdict.json
```

Platform-specific thresholds are defined in `.flagcicd/golden_baseline.json`:
- CUDA: 2.0% (strictest - primary platform)
- Ascend: 3.0%
- Other platforms: 3.0%

Exit code 1 indicates regression detected. Currently runs in **advisory mode** (non-blocking) to allow calibration period.

**Statistical** (3σ bounds) - planned:
```bash
python3 .github/scripts/flagcicd_regression.py \
  --current report.json \
  --historical historical.json \
  --sigma 3.0 \
  --output verdict.json
```

Statistical mode will be enabled after FlagOps API integration for historical data storage.

## Artifacts

- `flagcicd-report-{platform}-{scope}.json`: per-platform report (14-day retention)
- `flagcicd-run-summary.json`: aggregated run summary (14-day retention)

## Documentation

- CI testing guide: `tests/CI_TESTING_GUIDE.md`
- Entry workflow: `.github/workflows/flagcicd-entry.yml`
- Common workflows: `.github/workflows/all_tests_common.yml`, `.github/workflows/unit_tests_common.yml`
