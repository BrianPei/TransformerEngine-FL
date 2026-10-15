# Platform selection and CI gate

For pull requests to `main` and `devops`, `ci.yml` runs lint, computes the
platform matrix, runs the selected platform pipelines, and reports `CI Complete`.

Each platform config defines `paths_watch` using exact repository paths or
directory prefixes ending in `/`. Paths are case-sensitive; glob patterns are
not supported. Vendor aliases include `npu` for Ascend and `kunlunxin` for Kunlun.

- A platform-owned change selects that platform. Multiple platforms are combined.
- Every non-documentation path must match a rule. Shared or unknown paths run all
  platforms, including in PRs that also contain a platform-owned change.
- Changes under `docs/`, or to `.md` and `.rst` files, require no hardware tests.
  Lint and `CI Complete` still run for documentation-only PRs.
  This is an explicit routing policy, not a guarantee that these files can never
  affect runtime behavior. If tests or builds start consuming files in this
  category as inputs, update `_is_documentation` and its regression tests before
  relying on hardware-test skipping for those paths.
- Missing diff information or missing watch rules run all platforms. Invalid
  platform configurations fail matrix generation.
- Renames are treated as deletion plus addition to include both affected paths.
- Pushes that trigger CI, schedules, and manual runs select all platforms.

Keep shared dependencies out of individual platform watch lists: shared code
must continue to trigger all platforms. A new platform without watch rules
temporarily disables filtering rather than risking a missed test.

`CI Complete` requires successful lint and matrix generation. A nonempty matrix
requires successful platform pipelines. An empty matrix requires the platform
pipeline to be skipped. Failure, cancellation, or an unexpected skip fails the
gate. Inside each platform, tests may be skipped only when their workflow input
explicitly disables them; artifact builds may be skipped only when disabled in
the platform config.

After verifying a real PR run, configure the protected branch to require the
`CI Complete` status check instead of individual platform test checks. This is a
repository setting and is not changed by these files. Existing independent lint
and other workflow requirements should be reviewed separately.

Run the hardware-independent regression checks with:

```sh
python -m pip install PyYAML==6.0.3
python -m unittest discover -s .github/scripts/ci -p 'test_*.py'
```

Inspect routing without running hardware tests:

```sh
python .github/scripts/ci/generate_matrix.py \
  --event-name pull_request \
  --changed-files tests/plugin/backend/ascend/run_unit_tests.sh
```

## Validation on GitHub Actions

Local checks validate routing and execute the gate commands, including failure,
cancellation, and skip combinations. Running these checks on an accelerator
machine still does not validate how GitHub propagates results through nested
reusable workflows and matrices. Verify the following scenarios on GitHub before
changing required checks.

Use a test base branch that already contains this CI implementation, such as
`devops` (which is included in the PR trigger). Create separate scenario branches
and PRs against that base. The PR introducing the CI files itself selects the
full matrix, so it cannot demonstrate documentation-only or vendor-only routing.

| Scenario | Changes or action | Expected result |
| --- | --- | --- |
| Documentation only | Change only `README.rst` or a `.md` file | Lint and matrix generation succeed; matrix is `[]`; platform pipeline is skipped; `CI Complete` succeeds. |
| Vendor only | Change a watched vendor test script, with lint passing | Only the corresponding platform runs; its summary and `CI Complete` succeed. |
| Mixed vendor and shared code | Change a vendor file and a shared runtime file | All configured platforms are selected. |
| Test failure | Temporarily add `exit 1` to a watched vendor test script in the scenario branch | The test, its enabled test summary, platform workflow, and `CI Complete` fail. |
| Lint failure | Introduce a change rejected by an existing pre-commit hook | Lint fails; matrix and platform jobs are skipped; `CI Complete` runs and fails. |
| Cancellation | Cancel a running selected test job or the workflow run | The cancelled work must not produce a successful `CI Complete`; the gate fails if it runs, or remains cancelled if the whole run is cancelled. |

For each run, inspect the platform-selection summary, matrix jobs, nested test
summaries, and the final check on the PR's current commit. Keep intentional
failures in disposable scenario branches; do not merge them into the test base.
Only switch branch protection to `CI Complete` after these scenarios have been
observed. No live Actions results are claimed by the local regression suite.
