# TransformerEngine-FL CI

The active entries are `ci.yml` (PR, push to main, manual dispatch) and
`nightly.yml` (02:17 Asia/Shanghai, or manual dispatch). Both generate a plan
and use the shared platform and group workflows. `CI Complete` and
`Nightly Complete` aggregate the results.

## Selection and levels

PR routing uses platform `paths_watch`. Vendor paths select their platforms;
shared/unknown paths and unavailable diffs select all platforms. Documentation
only changes skip hardware. `platform:<name>` labels override routing, so a
narrowed shared-code run requires reviewer judgment. `level:L0` through
`level:L4` labels can escalate the inferred level. Risk rules also raise it.

PR defaults to L1; branch/CI dispatch defaults to L2; Nightly is fixed at L3.
Levels are cumulative, subject to group path filters and platform support.
Nightly has no diff filters and runs all currently executable configured groups.
L1 unit launchers also include distributed/ONNX work and declare multi-device
resources. Integration is L2. FA compatibility and benchmarks are L3 but are
currently blocked by runner/baseline requirements. L4 acceptance groups remain
blocked by explicit contract gaps. See `.flagcicd/project.yaml`.

The planner supports `manual` and `release` stages via CLI. Dedicated manual
selection and release-acceptance workflows are not implemented. Deleted comment
trigger workflows are not active; the retained request parser alone does not
provide `/ci run` support. Weekly is not configured.

## Execution and evidence

The plan pins source SHA, platform config hashes, group commands, resources,
and coverage settings. Each platform slices it into parallel and serial lanes.
Exclusive or multi-device groups run serially after parallel work. This only
orders jobs within a platform invocation; runner deployment must provide
physical device isolation across PR/Nightly runs.

Optional wheel artifacts carry identity metadata checked before setup. Group
reports contain plan/source identity, status, execution and duration. Required
groups are determined from the plan and must pass with execution evidence.
Missing, duplicate, mismatched or failed required reports fail the platform gate.
Missing or corrupt JUnit cannot be interpreted as passed. Preflight checks
repository structure/Python syntax; it does not certify device availability.
The global gate requires successful planning/pipeline results and complete
platform reports. Required blocked groups fail acceptance.

## Coverage

Coverage measures Python source, not C++ or device kernels. Raw hidden coverage
fragments are uploaded by group jobs, then combined in `all_tests_common.yml`.
Platforms with `coverage.required: true` fail if collection/aggregation fails
or the coverage job is skipped. This requirement is separate from regression
thresholds, which are advisory.

PR coverage compares against a main-branch artifact of the same platform/device.
Nightly uses a 14-run history with at least 8 samples and a lower bound of
mean minus three standard deviations. Threshold configuration lives in
`.flagcicd/golden_baseline.json`. The current historical download step still
requires `gh` in vendor images; availability and live baseline collection must
be validated on runners. FlagOps uploads are advisory and need
`FLAGCICD_API_TOKEN`; fork runs usually have no secret.

## Validation

```sh
python -m unittest discover -s .github/scripts/ci -p 'test_*.py'
python .github/scripts/ci/flagcicd_planner.py --stage nightly --output /tmp/plan.json
```

Local validation does not prove hardware compatibility or GitHub nested matrix
behavior. Validate docs-only/vendor/shared routing, deliberate test/setup
failure, absent reports, coverage failure and cancellation on Actions before
changing required checks. Configure `CI Complete` in branch protection manually.
Schedules require the workflow on the default branch and may be delayed.
