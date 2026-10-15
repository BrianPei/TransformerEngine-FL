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

### Optional routing and retries

- `platform:<config-name>` PR labels override path routing; multiple labels take
  their union, and `platform:all` selects all platforms. Unknown platform labels
  fail generation. Adding/removing labels reruns CI; removing all restores path
  routing. Non-PR runs are unaffected. Narrow shared-code changes only after review.
- Unit/integration groups retry up to twice, five seconds apart, within the same
  job timeout. Each attempt is logged; exhaustion fails the job. Signal exits are
  not retried. A group can set `max_retries: 0`, `1`, or `2`; outside CI the default
  is zero. Setup, builds and lint are not retried.
- Writers/maintainers/admins can comment `/ci run ascend`, `/ci run cuda,ascend`,
  `/ci run all`, or `/test musa` on an open PR. `ci-requested.yml` checks live
  permissions and passes the current head SHA through all configuration/build/
  test/coverage checkouts, including fork PRs. Authorization uses trusted base code
  and a read-only token; requested PR code executes on selected self-hosted hardware.
  Invalid commands/platforms, issue/edited comments, closed PRs and unauthorized
  users cannot start tests. `Requested CI Complete` is supplemental to required
  PR CI. Comment triggering requires this workflow on the default branch and uses
  the GitHub CLI bundled with ubuntu-latest; existing manual workflows are unchanged.

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

## FlagCICD coverage and performance reports

Coverage-enabled platforms collect Python coverage per unit-test group, combine
raw data for groups named `pytorch_*`, and upload
`coverage-<platform>-<device>-pytorch-unittest.json` to GitHub artifacts and
`http://flagcicd-inner.flagos.net:8000/metrics/` with `file_type: coverage`.
This does not measure C++ or device-kernel source coverage.

Configure `FLAGCICD_API_TOKEN` as a repository or organization Actions secret
accessible to this repository. Unified CI, manual platform workflows, and
comment requests forward it explicitly through reusable workflows. No user ID
or `x-user-id` header is sent, per the service owner's integration instructions.
The public FlagOps `post-pytest-report@v2` still rejects empty user IDs, so
`.github/scripts/ci/upload_report.py` sends its multipart form protocol directly
using token authentication. It also attaches repository/workflow/job/run/runner
metadata and the PR number for pull-request and issue-comment events. The
backend job name is derived from the report filename without `.json`.

The uploader bypasses proxies, limits connection time to 10 seconds and the
whole request to 60 seconds, and never retries POSTs automatically. Missing
tokens/files and upload failures are warnings. Reports remain available as
GitHub artifacts when artifact upload succeeds; HTTP delivery status appears in
the job summary. A 2xx response does not establish successful report parsing.

The reporting containers use normal DNS by default. If a runner network needs
an explicit hosts entry, configure the repository/organization Actions variable
`FLAGCICD_HOST_IP` with an IP reachable from that network. Both reporting
workflows then add `--add-host flagcicd-inner.flagos.net:<IP>` to the container;
no host machine changes are needed. The service team supplied `10.1.4.15`, but
local validation on the Enflame development machine timed out on that IP/port
8000. Its existing DNS route returned HTTP 404 for GET `/metrics/`, which only
establishes HTTP connectivity, not a valid upload endpoint. Verify the address
and authenticated endpoint in the actual runner network before setting the
variable. An `/etc/hosts` entry supplies name resolution, not network routing.

### Performance smoke

All eight platform configs enable `performance_smoke`: Ascend, CUDA, Enflame,
Hygon, KunlunXin, MetaX, MUSA, and PPU. Every configured device runs the shared
suite with its platform image, setup script and backend selection. The template
keeps the suite disabled until a new platform is configured.
Ascend reuses its existing NPU test compatibility patch. Device aliases exposed
by vendor CUDA compatibility layers are resolved using an actual device tensor,
so TE input validation and report metadata use the runtime device type.
The suite waits for enabled unit and integration tests to succeed, reuses the
platform image/setup/build artifact, and runs six BF16 cases: Linear, LayerNorm,
and RMSNorm, each in forward and forward/backward mode, with input shape
`[256, 512]`. It does not retry measurements or collect coverage.

Each case performs 10 warmup iterations and 30 measured iterations. The timer
uses host wall time with accelerator synchronization after each operation;
reported latency includes host dispatch, and backward measurements include
clearing gradients. These are end-to-end smoke measurements, not isolated kernel
times. A finite output/gradient check precedes measurement. Missing hardware,
failed cases, or an empty suite fail the job. Ordinary Python exceptions still
produce a report; cancellation, process termination, or a step timeout may not.

The JSON artifact `perf-smoke-<platform>-<device>.json` uses TE schema version 1:
`environment` records the tested commit, image, software and device;
`measurement` describes timing; `cases` records operator/mode/shape/dtype/status
and `latency` samples/min/median/max in milliseconds; `summary` records totals
and overall status. Logs and a latency table in the job summary aid inspection.
No performance regression threshold is enforced yet. Enabled smoke jobs must
pass the final platform gate; disabled jobs must be skipped.

The backend performance schema has not been supplied. Uploads provisionally use
`file_type: performance` by default; a platform can override it with
`performance_smoke.file_type`. Confirm
this identifier and parsing against a backend example; adapt the JSON if needed.
Do not interpret a 2xx response as validated performance ingestion.

### First live verification

1. Run each `all_tests_<platform>.yml` on the branch containing these changes,
   or run unified CI for all platforms. Each runs configured unit/integration
   tests, coverage aggregation, then performance smoke. Validate every configured
   device separately; MetaX C500 filenames below are examples.
2. Inspect `coverage-metax-c500-pytorch-unittest.json`. Missing coverage
   dependencies/data can prevent generation with `coverage.required: false`.
3. Inspect `perf-smoke-metax-c500.json`: six passed cases, finite positive latency
   samples, correct tested commit/device/image, and matching CI summary.
4. Check both uploader summaries for HTTP 2xx. The aggregation and smoke
   containers must resolve and directly reach the internal metrics endpoint.
5. Verify repository/run/job ownership, coverage totals/file details, and
   performance metrics in FlagCICD. Request the performance schema from the
   service owner if the provisional report is rejected or not parsed.

Local tests cover the HTTP form using a loopback server, error handling, report
failure output, timing validation and workflow gates. They do not establish a
live authenticated upload or accelerator performance. Do not publish results
from CPU-only/mocked tests as device measurements.

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
| Label routing | Add/remove `platform:ascend`, combine vendor labels, then use `platform:all` | Each label change reruns PR CI with the documented override; removing the labels restores path routing. |
| Retry recovery/exhaustion | Use a disposable test group that fails once then passes; separately make it always fail | Attempt counts are visible; recovery succeeds, while exhaustion fails summaries and the final gate. |
| Authorized comment | A writer comments `/ci run ascend` on an open PR | Only Ascend runs; every checkout uses the captured PR SHA; `Requested CI Complete` reports the outcome. |
| Rejected comment | Use a read-only account, an invalid platform, an issue comment, or a closed PR | No hardware tests start. |
| Pinned/fork PR request | Update the PR after resolution; separately request a fork PR | Every checkout still tests the captured SHA using base-repository workflows. |

For each run, inspect the platform-selection summary, matrix jobs, nested test
summaries, and the final check on the PR's current commit. Keep intentional
failures in disposable scenario branches; do not merge them into the test base.
Only switch branch protection to `CI Complete` after these scenarios have been
observed. No live Actions results are claimed by the local regression suite.
