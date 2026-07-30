# TransformerEngine-FL Tests

This document describes how tests are organized, how the shared CI workflow
selects and runs them, how to add a test, and how to reproduce a CI test group
on a development machine.

## Test Ownership

Keep a test close to the code boundary it validates:

```text
tests/
|-- cpp/                         Upstream Transformer Engine C++ tests
|-- cpp_distributed/             Upstream distributed C++ tests
|-- jax/                         Upstream JAX tests
|-- pytorch/                     Upstream Transformer Engine PyTorch tests
|-- plugin/
|   |-- plugin/                  Plugin manager, policy, and discovery tests
|   |-- backend/
|   |   |-- reference/           Reference backend-specific behavior
|   |   |-- flagos/              FlagOS backend-specific behavior
|   |   `-- npu/                 NPU device tests and upstream-test adapter
|   |-- conftest.py              Shared plugin fixtures
|   |-- run_upstream.py          Runs upstream tests through an adapter
|   `-- utils.py                 Small shared plugin test utilities
`-- test_utils/
    `-- run_ci_test_group.py      Generic CI matrix-group runner
```

Use the following rules when choosing a location:

- Keep upstream TE tests in `tests/cpp`, `tests/jax`, or `tests/pytorch`.
- Put plugin registration, policy, and manager tests in `tests/plugin/plugin`.
- Put behavior required from one backend in that backend's directory under
  `tests/plugin/backend`.
- Put real-device bootstrap, compatibility adapters, and device-only tests in
  the corresponding device directory, such as `tests/plugin/backend/npu`.
- Do not add platform branches to an upstream test only to make CI start. Use a
  platform adapter or platform test script instead.

See `tests/plugin/README.md` for additional plugin test ownership notes.

## CI Execution Model

Each platform uses the same workflow implementation:

```text
.github/workflows/all_tests_<platform>.yml
  -> .github/workflows/all_tests_common.yml
  -> .github/configs/<platform>.yml
  -> .github/workflows/unit_tests_common.yml
  -> .github/scripts/setup_<platform>.sh
  -> tests/test_utils/run_ci_test_group.py
```

The platform entry workflow only selects a platform and enables unit or
integration tests. `all_tests_common.yml` reads the platform configuration and
creates one job for every `device_types` and `unit_test_matrix` combination.
The common unit workflow starts the configured image, exports `build_env`, runs
the setup script, and delegates the selected matrix group to
`run_ci_test_group.py`.

Platform-specific behavior belongs in these files:

| Concern | Source of truth |
| --- | --- |
| Runner labels, image, devices, mounts, and container options | `.github/configs/<platform>.yml` |
| Runtime paths, Python environment, installation, and preflight checks | `.github/scripts/setup_<platform>.sh` |
| Test targets, arguments, environment, and JUnit names | `unit_test_matrix` in the platform config |
| Complex platform-specific selection or skip policy | A platform-owned script referenced by the config |
| Shared matrix execution and coverage | Common workflows and `tests/test_utils/run_ci_test_group.py` |

Adding a chip should not require a chip-name conditional in a common workflow.
Start from `.github/configs/template.yml` and add a setup script and a small
platform entry workflow.

## Adding a Test

### 1. Add and run the test directly

Use pytest-compatible names and assertions so collection failures are visible:

```bash
python3 -m pytest -v tests/plugin/plugin/test_manager.py
python3 -m pytest -v tests/plugin/backend/reference/test_gemm.py
python3 -m pytest -v tests/pytorch/test_sanity.py -k test_sanity_linear
```

Several plugin tests install mock modules in `sys.modules`. Keep unrelated
plugin test files in separate pytest steps or processes when combining them
would make execution order-dependent.

### 2. Add the test to a CI group

There are two supported group runners.

#### Pytest group

Use `runner: pytest` when a group can be expressed as pytest targets and
arguments:

```yaml
unit_test_matrix:
  - name: pytorch_unittest
    runner: pytest
    log_dir: logs/L0_pytorch_unittest-example
    pytest_args: ['-v', '-s', '--tb=short']
    env:
      NVTE_FUSED_ATTN: '0'
    steps:
      - name: reference GEMM
        junit: pytest_reference_gemm.xml
        targets:
          - tests/plugin/backend/reference/test_gemm.py
      - name: selected upstream linear test
        junit: pytest_linear.xml
        targets:
          - 'tests/pytorch/test_sanity.py::test_sanity_linear[False-False-False-small-None-dtype0]'
        env:
          NVTE_FLASH_ATTN: '0'
```

Supported fields are:

| Field | Meaning |
| --- | --- |
| `name` | Stable matrix and job name |
| `runner` | `pytest` or `script` |
| `log_dir` | Directory for generated JUnit XML files |
| `pytest_args` | Arguments shared by every step in the group |
| `env` | Environment shared by every step in the group |
| `steps[].targets` | Test files, directories, or exact pytest node IDs |
| `steps[].args` | Additional pytest arguments for one step |
| `steps[].env` | Environment overrides for one step |
| `steps[].junit` | JUnit output filename |
| `steps[].requires_modules` | Python modules that must import before the step runs |
| `steps[].use_platform_runner` | Set to `false` to bypass a platform pytest adapter |

The command configured by `TE_TEST_PYTEST_COMMAND` is used by default. For
example, Ascend routes upstream tests through the adapter in
`tests/plugin/backend/npu`. Backend-independent plugin tests can opt out with
`use_platform_runner: false`.

#### Script group

Use `runner: script` when an existing QA script owns selection, dependency
checks, multiple subprocesses, or platform skip policy:

```yaml
unit_test_matrix:
  - name: pytorch_unittest
    runner: script
    path: qa/plugin/example/test.sh
    args: [unittest]
    env:
      XML_LOG_DIR: logs/L0_pytorch_unittest-example
```

The generic runner executes `bash <path> <args...>` from the repository root.
Keep chip-specific branches in that platform-owned script, not in
`unit_tests_common.yml`.

### 3. Verify collection and failure behavior

Before opening a PR, verify the exact configured target:

```bash
python3 -m pytest --collect-only -q path/to/test_file.py
python3 -m pytest -q -x path/to/test_file.py
```

A test file that reports `0 collected` is not coverage. Do not treat pytest
exit code 5 as a successful test run unless the group is explicitly optional
and that behavior is documented.

## Adding a Platform

1. Copy `.github/configs/template.yml` to
   `.github/configs/<platform>.yml`.
2. Set the runner labels and one or more `device_types`.
3. Set `ci_image` to an image containing the compiler, framework, and device
   runtime required by the tests.
4. Add required host mounts and device flags to `container_volumes` and
   `container_options`.
5. Add `.github/scripts/setup_<platform>.sh`. It should activate Python, load
   the accelerator runtime, make TE-FL importable through installation or
   `PYTHONPATH`, verify the device, and append any environment needed by later
   steps to `$GITHUB_ENV`.
6. Define `unit_test_matrix` with `pytest` or `script` groups. Use an empty
   `integration_test_matrix` when integration tests are disabled.
7. Add `.github/workflows/all_tests_<platform>.yml` that calls
   `all_tests_common.yml` with the platform name.
8. Reproduce each group in the configured image before enabling the workflow.

GitHub displays a manually dispatched workflow only after its entry workflow
exists on the repository's default branch. Register the entry there first,
then select the development branch as the ref when dispatching its tests.

Do not modify the common workflows merely to recognize the new platform. A
common-workflow change is appropriate only when it adds a capability useful to
all platforms.

## Running Tests Locally

### Fast path in an existing environment

If TE-FL and the accelerator runtime are already installed, run a single test
or a platform-owned script directly:

```bash
export TE_PATH="$PWD"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XML_LOG_DIR="$PWD/logs"
mkdir -p "$XML_LOG_DIR"

python3 -m pytest -v tests/plugin/plugin/test_policy.py
bash qa/L0_pytorch_unittest/test.sh
bash qa/plugin/hygon/test.sh unittest
```

For an upstream test that needs the Ascend compatibility adapter:

```bash
bash tests/plugin/backend/npu/run_native.sh \
  -v -s tests/pytorch/test_sanity.py -k test_sanity_linear
```

### Use the same image as CI

Run on a host with the target accelerator driver and Docker support. The
platform config is the source of truth for the image and container flags:

```bash
PLATFORM=ascend                       # cuda, metax, ascend, or hygon
CONFIG=".github/configs/${PLATFORM}.yml"
IMAGE="$(yq -r '.ci_image' "$CONFIG")"

echo "$IMAGE"
docker pull "$IMAGE"
```

Start an interactive container with:

- every volume in `container_volumes` converted to `-v host:container`;
- the flags in `container_options`;
- the repository mounted into the container.

The general form is:

```bash
docker run --rm -it \
  <container volume arguments> \
  <container options> \
  -v "$PWD:/workspace/TransformerEngine-FL" \
  -w /workspace/TransformerEngine-FL \
  "$IMAGE" bash
```

Do not omit device nodes, driver mounts, runtime selection, or groups from the
platform config. A container that imports PyTorch but cannot see the device is
not equivalent to CI.

### Reproduce one complete CI group inside the container

The following commands reproduce the environment setup and matrix runner. They
require Mike Farah `yq` v4, the same parser used by the workflow:

```bash
cd /workspace/TransformerEngine-FL

PLATFORM=ascend
TEST_GROUP=pytorch_unittest
CONFIG=".github/configs/${PLATFORM}.yml"

export GITHUB_WORKSPACE="$PWD"
export GITHUB_ENV=/tmp/transformer-engine-github-env
: > "$GITHUB_ENV"

# Export build_env from the platform configuration.
while IFS=$'\t' read -r key value; do
  export "$key=$value"
done < <(
  yq -r \
    '.build_env // {} | to_entries[] | [.key, (.value | tostring)] | @tsv' \
    "$CONFIG"
)

# Activate/install/verify the platform environment.
SETUP_SCRIPT="$(yq -r '.setup_script // ""' "$CONFIG")"
if [ -n "$SETUP_SCRIPT" ]; then
  bash "$SETUP_SCRIPT"
fi

# setup_<platform>.sh persists values here for subsequent CI steps.
set -a
. "$GITHUB_ENV"
set +a

export TE_PATH="$PWD"
export TE_LIB_PATH="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs

export TEST_GROUP
export TE_TEST_GROUP_JSON="$(
  yq -o=json -I=0 \
    '.unit_test_matrix[] | select(.name == strenv(TEST_GROUP))' \
    "$CONFIG"
)"

test -n "$TE_TEST_GROUP_JSON"
python3 tests/test_utils/run_ci_test_group.py
```

Change only `PLATFORM` and `TEST_GROUP` to reproduce another configured group.
The group names are listed under `unit_test_matrix` in the selected config.
This path creates the same JUnit files as CI. Coverage collection is added by
`unit_tests_common.yml`; use the workflow when validating the complete coverage
upload and aggregation path.

## Common Failures

- **Image pulls but imports fail:** run the platform setup script; the image is
  the base environment, while TE-FL is normally installed from the checkout.
- **Device is unavailable:** compare the local Docker mounts and options with
  the platform config and verify the host driver independently.
- **A test passes directly but fails in CI:** reproduce the complete matrix
  group because setup, adapter, group environment, and execution order matter.
- **A plugin test fails only in a combined run:** run each file in a fresh
  process and check for mocked modules left in `sys.modules`.
- **No tests are collected:** fix test naming or the configured target; do not
  convert exit code 5 into success.
- **Dependencies are missing:** add them to the CI image when they are stable
  platform requirements, or install and verify them in the platform setup
  script. Do not install chip-specific packages in a common workflow.
