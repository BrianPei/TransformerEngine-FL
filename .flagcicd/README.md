# FlagCICD Contract v1

TransformerEngine-FL CI/CD execution framework with L0-L4 test levels and differentiated PR-vs-Nightly scheduling.

## Schema Versions

All schemas use **v1** (internal version, not formally released):

- `flagcicd.project/v1` - Project configuration contract
- `flagcicd.plan/v1` - Execution plan
- `flagcicd.group-report/v1` - Single test group report
- `flagcicd.platform-report/v1` - Platform-aggregated report
- `flagcicd.gate/v1` - Global gate result
- `flagcicd.artifact-identity/v1` - Build artifact identity
- `flagcicd.report/v1` - Coverage report (legacy compatibility)

## Core Files

### Configuration
- `project.yaml` - Project-level config: stages/levels/resources/test group catalog
- `golden_baseline.json` - Coverage baseline and regression thresholds
- `../.github/configs/*.yml` - Platform configs: images/runners/matrices

### Python Modules (../.github/scripts/ci/)
- `flagcicd_contract.py` - Schema constants and validation (stdlib only)
- `flagcicd_planner.py` - Execution plan generator
- `flagcicd_gate.py` - Four subcommands: slice/group-report/platform/gate
- `flagcicd_identity.py` - Artifact identity management
- `test_planner.py` - Planner unit tests

### GitHub Actions (../.github/actions/)
- `flagcicd-plan/` - Plan generation composite action
- `flagcicd-gate/` - Gate validation composite action

### Workflows (../.github/workflows/)
- `ci.yml` - PR workflow: lint -> plan -> platform-pipeline -> ci-complete
- `nightly.yml` - Nightly workflow: lint -> plan -> platforms -> complete
- `_platform_pipeline.yml` - Platform-level pipeline (reusable)
- `all_tests_common.yml` - Per-platform test execution (reusable)
- `test_groups_common.yml` - Test group matrix execution (reusable)

## Test Levels

### L0: Preflight
- Quick validation (linting, fast unit tests)
- Resource: `shared_cpu`
- Stage: PR (when enabled), branch

### L1: Unit Tests
- Fast isolated tests
- Resources: `shared_cpu`, `single_device`
- Stage: PR (default), branch

### L2: Integration Tests
- Cross-component integration
- Resources: `single_device`, `multi_device`
- Stage: PR (when requested), branch

### L3: Nightly
- FlashAttention version compatibility tests
- Benchmark regression tests
- Resources: all except `release_exclusive`
- Stage: nightly

### L4: Release
- Full release validation
- Resources: all including `release_exclusive`
- Stage: release

## Resource Classes

Resource classes determine scheduling lanes:

### Parallel Lane (max-parallel=256)
- `shared_cpu` - CPU-only tests, no GPU required
- `single_device` - Single GPU per test
- `multi_device` - Multiple GPUs per test (typically 2 or 4)
- `communication_exclusive` - Needs exclusive communication channels

### Serial Lane (max-parallel=1)
- `model_exclusive` - Large models requiring exclusive node
- `hopper_only` - Hopper-specific tests (e.g., FP8 attention)
- `release_exclusive` - Release-only validation

## Stages

### pr
- Selection: `affected` (changed files + labels)
- Default level: L1 (unit tests)
- Coverage: regression check against main branch

### branch
- Selection: `all` platforms
- Default level: L1
- Triggered: push to main

### nightly
- Selection: `all` platforms
- Default level: L3 (includes FA versions)
- Coverage: 3-sigma check against 14-run rolling window

### manual
- Selection: user-specified platforms
- Level: user-specified or auto
- Triggered: workflow_dispatch

### release
- Selection: `all` platforms
- Default level: L4
- Blocked groups: excluded from L1-L3

## Plan Hash

The plan hash is SHA256 of the canonical JSON representation (deterministic, excludes `plan_hash` field itself):

```python
canonical_json = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
plan_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
```

## Artifact Identity

Each test artifact includes `artifact-identity.json`:

```json
{
  "schema_version": "flagcicd.artifact-identity/v1",
  "platform": "cuda",
  "source_sha": "abc123...",
  "config_hash": "def456...",
  "wheels": ["transformer_engine-1.0.0-cp310-cp310-linux_x86_64.whl"]
}
```

Identity verification:
- `platform` must match platform config name
- `source_sha` must match plan's source_sha
- `config_hash` must match platform config hash
- Failures block test execution

## Gate Logic

### Group Report (`flagcicd_gate.py group-report`)
Input: step outcomes + JUnit XML
Output: `flagcicd-group-{kind}-{device}-{gid}.json`

Status determination:
- All steps successful + all tests passed -> `passed`
- Any step failed + tests exist -> `failed`
- No tests collected -> `no_tests`
- All skipped -> `skipped`

### Platform Report (`flagcicd_gate.py platform`)
Input: all group reports for the platform
Output: `flagcicd-platform-{platform}.json`

Validation:
- All required groups must have status `passed`
- Optional groups can be `passed`, `skipped`, or `no_tests`
- Any `failed` or missing required groups -> platform fails

### Top-Level Gate (`flagcicd_gate.py gate`)
Input: plan + needs JSON
Output: exit code (0=pass, 1=fail)

Requirements:
- Plan job: `success`
- Pipeline job: `success`
- All platform reports: status `passed`
- Lint-only plans: skip pipeline validation

## Coverage Strategy

### PR (absolute threshold)
Compare PR coverage against main branch baseline:
- Line coverage must not decrease by more than `pr_regression_threshold` (default: 1.0%)
- Advisory only (does not block CI)

### Nightly (3-sigma)
Compare nightly coverage against rolling 14-run window:
- Compute mean and stddev from last 14 nightly runs
- Current coverage must be within mean - 3*stddev
- Failing 3-sigma triggers investigation

## Test Group Catalog

10 groups defined in `project.yaml`:

1. **pytorch_distributed** - Multi-GPU distributed tests
2. **pytorch_fused_attn** - Fused attention tests (CUDA only)
3. **pytorch_core** - Core PyTorch layer tests
4. **pytorch_basic** - Basic functionality tests
5. **jax** - JAX integration tests
6. **cpp** - C++ unit tests
7. **docs** - Documentation build
8. **integration_common** - Cross-plugin integration (common paths)
9. **integration_compat** - Compatibility tests (torch 2.0, 2.1, 2.2, 2.5)
10. **fa_versions** - FlashAttention version matrix (L3 only)

Each group specifies:
- `kind`: unit/integration/compat
- `description`
- `resource_class`
- `lane`: parallel/serial
- `run`: command template
- `levels`: allowed test levels
- `coverage`: whether to collect coverage (default: false)

## Example: PR Flow

1. Developer pushes to PR branch
2. CI triggers with `GITHUB_EVENT_NAME=pull_request`
3. **Lint** job runs on PR diff
4. **Plan** job (`flagcicd-plan` action):
   - Detects changed files
   - Selects affected platforms (e.g., `cuda`, `ascend`)
   - Generates plan with L1 groups
   - Outputs plan JSON + plan_hash
5. **Platform-pipeline** job (matrix: 2 platforms):
   - Each platform calls `all_tests_common.yml`
   - `checkout_and_config` slices plan into parallel/serial matrices
   - `build_test_artifact` creates wheel + identity
   - `parallel_groups` runs 8 groups concurrently
   - `serial_groups` runs 0 groups (no serial in L1)
   - `coverage` aggregates fragments, checks regression
   - `platform_gate` validates all required groups passed
6. **CI-complete** job (`flagcicd-gate` action):
   - Downloads all platform reports
   - Validates plan + pipeline + platforms
   - Exits 0 (CI green) or 1 (CI red)

## Example: Nightly Flow

1. Scheduled trigger at 00:00 UTC
2. **Lint** job runs (trivial pass on identical diff)
3. **Plan** job with `--stage nightly`:
   - Selects all 8 platforms
   - Generates plan with L3 groups (includes `fa_versions`)
4. **Platforms** job (matrix: 8 platforms):
   - Each platform runs all L3 groups
   - `fa_versions` group tests FA 2.5.7, 2.5.8, 2.5.9, 2.6.1, 2.6.3
   - Coverage checked against 3-sigma
5. **Complete** job validates all platforms

## Troubleshooting

### Plan generation fails
- Check `project.yaml` syntax (YAML)
- Verify platform configs exist in `.github/configs/`
- Check `--stage` value is valid

### Group report status = failed
- Download group artifact: `flagcicd-group-{kind}-{device}-{gid}.json`
- Check `steps` object for failed step
- Check `junit_summary` for test failures

### Platform gate fails
- Download platform report: `flagcicd-platform-{platform}.json`
- Check `groups` array for failed/missing required groups
- Check `validation.required_missing` for details

### Top-level gate fails
- Download all platform reports from artifacts
- Run locally: `python .github/scripts/ci/flagcicd_gate.py gate --plan plan.json --needs needs.json`
- Check stderr for validation errors

## Migration from Old CI

### Removed workflows
- `unit_tests_common.yml` -> `test_groups_common.yml` (kind=unit)
- `integration_tests_common.yml` -> `test_groups_common.yml` (kind=integration)

### Removed scripts
- `aggregate_flagcicd_reports.py` -> `flagcicd_gate.py platform`
- `nightly_report.py` -> `flagcicd_gate.py gate`
- `flagcicd_plan.py` -> `flagcicd_planner.py`

### Key differences
- Plan-driven execution (deterministic, reproducible)
- Unified group execution (no unit/integration split)
- Resource-based lane scheduling (parallel vs serial)
- Artifact identity verification (prevents stale builds)
- Structured JSON reports (machine-readable)

## References

- Project config: `.flagcicd/project.yaml`
- Platform configs: `.github/configs/*.yml`
- Contract module: `.github/scripts/ci/flagcicd_contract.py`
- Test suite: `.github/scripts/ci/test_planner.py`, `test_ci.py`
