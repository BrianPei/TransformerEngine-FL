# TransformerEngine-FL CI

The active entries are `ci.yml` (PR, push to main, manual dispatch),
`daily.yml` (stage `nightly`, 02:17 Asia/Shanghai, or manual dispatch),
`ci-requested.yml` (stage `manual`) and `release-acceptance.yml` (stage
`release`). Each generates a plan and uses the shared platform and group
workflows. `CI Complete`, `Daily Complete`, `CI Requested Complete` and
`Release Acceptance Complete` gate the results.

## Selection and levels

PR routing uses platform `paths_watch`. Vendor paths select their platforms;
shared/unknown paths and unavailable diffs select all platforms. Documentation
only changes skip hardware. `platform:<name>` labels override routing, so a
narrowed shared-code run requires reviewer judgment. `level:L0` through
`level:L4` labels can escalate the inferred level. Risk rules also raise it.

PR defaults to L1; branch/CI dispatch defaults to L2; Daily is fixed at L2 with default risk R2.
Each changed file takes the highest level of the risk rules it matches, or
`unknown_level` when it matches none; the plan takes the maximum over files.
Levels are cumulative, subject to group path filters and platform support.
Daily has no diff filters and runs all currently executable configured L0-L2 groups.
Groups with `watch_platform_paths` also watch the platform `paths_watch` and
their matrix entry `path`, so a launcher-only PR runs the group it launches.
Plans record per-platform `reasons`, `common_path_triggers` and an `excluded`
ledger of in-level groups dropped by path filters or missing matrix entries.
L1 unit launchers also include distributed/ONNX work and declare multi-device
resources. Integration is L2. FA compatibility and benchmarks are L3 but are
currently blocked by runner/baseline requirements. L4 acceptance groups remain
blocked by explicit contract gaps. See `.flagcicd/project.yaml`.

Requested runs: a PR comment `/ci run <platforms|all>` or `/test <platforms|all>`
from a user with write access runs the PR head SHA (GitHub only runs
`issue_comment` workflows from the default branch). `workflow_dispatch` on
`ci-requested.yml` takes platforms, a test level and a checkout ref.
`release-acceptance.yml` runs the fixed L4 release plan for one ref; required
blocked groups fail it. Weekly is not configured.

## Execution and evidence

The plan pins source SHA, platform config hashes, group commands, resources,
and coverage settings. Each platform slices it into parallel and serial lanes.
Exclusive or multi-device groups run serially after parallel work finishes,
whatever its result; the platform gate judges both lanes. This only
orders jobs within a platform invocation; runner deployment must provide
physical device isolation across PR/Nightly runs.

Optional wheel artifacts carry identity metadata checked before setup. Group
reports contain plan/source identity, status, execution and duration. Required
groups are determined from the plan and must pass with execution evidence.
Missing, duplicate, mismatched or failed required reports fail the platform gate.
Each group declares `evidence`: `junit` groups need non-empty JUnit results,
`exit_code` groups (preflight, script integration) pass on exit code 0.
Missing or corrupt JUnit cannot be interpreted as passed. Preflight checks
repository structure/Python syntax; it does not certify device availability.
The global gate requires successful planning/pipeline results and complete
platform reports. Required blocked groups fail acceptance.

## Coverage

Coverage measures Python source, not C++ or device kernels. Group jobs collect
the platform `coverage.sources` with `--cov-append`, upload raw hidden fragments,
and `all_tests_common.yml` combines them, applying `coverage.include`/`omit`.
Platforms with `coverage.required: true` fail if collection/aggregation fails
or the coverage job is skipped. This requirement is separate from regression
thresholds, which are advisory.

PR coverage compares against a main-branch artifact of the same platform/device.
Nightly uses a 14-run history with at least 8 samples and a lower bound of
mean minus three standard deviations. Threshold configuration lives in
`.flagcicd/golden_baseline.json`. Baselines are fetched with the Actions REST
API from Python, not `gh`; live baseline collection must be validated on
runners. FlagOps uploads are advisory and need
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

## Scope and device counts

This pipeline uses static routing: changed paths, stage defaults, risk rules and
labels select platforms and test groups. Runtime-evidence collection, cached
result pruning, priority waves and reduced workloads are outside this change.
JUnit/exit-code execution evidence, artifact identity checks and all three report
levels remain required by the existing plan and gate contracts.

Keep `nproc_per_node` in each `.github/configs/<platform>.yml`: it is a positive
integer device budget per multi-device test job, not the total number of cards
installed on the host. The planner rejects missing or invalid counts rather than
assuming one device. `multi_device` and exclusive resource classes resolve this
platform value; `single_device` groups still request one device. Counts may differ
by platform (currently 2, 4 or 8) and do not require duplicated workflows.

The budget is a scheduling declaration, not automatic hardware detection or a
universal torchrun override. Keep it aligned with container-visible devices and
platform launchers; launchers can use fewer devices in individual phases. Changing
physical card allocation also requires checking the runner, device mounts and
visibility settings. Runtime device isolation remains a runner responsibility.
