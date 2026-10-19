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
   - `test_level`: `L0` (preflight), `L1` (unit), `L2` (unit+integration)
   - `risk_level`: `auto` (uses risk mapper) or explicit `R1`-`R4`

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

Regression detection supports two modes:

**Absolute threshold** (default 2.0%):
```bash
python3 .github/scripts/flagcicd_regression.py \
  --current report.json \
  --baseline baseline.json \
  --threshold 2.0 \
  --output verdict.json
```

**Statistical** (3σ bounds):
```bash
python3 .github/scripts/flagcicd_regression.py \
  --current report.json \
  --historical historical.json \
  --sigma 3.0 \
  --output verdict.json
```

Exit code 1 indicates regression detected.

## Artifacts

- `flagcicd-report-{platform}-{scope}.json`: per-platform report (14-day retention)
- `flagcicd-run-summary.json`: aggregated run summary (14-day retention)

## Documentation

- CI testing guide: `tests/CI_TESTING_GUIDE.md`
- Entry workflow: `.github/workflows/flagcicd-entry.yml`
- Common workflows: `.github/workflows/all_tests_common.yml`, `.github/workflows/unit_tests_common.yml`
