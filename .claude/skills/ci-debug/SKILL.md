---
name: ci-debug
description: Debug or reason about the TransformerEngine-FL GitHub Actions CI — a failed/queued workflow run, a PR that ran the wrong platforms, plan or gate errors, coverage and Feishu notification problems, or adding/changing a platform. Use whenever the user mentions CI, workflow runs, the platform matrix, L0-L4 test levels, runner labels, FlagCICD, or a red check on this repository.
---

# CI Debug (TransformerEngine-FL)

This repository runs a FlagCICD v1 pipeline over eight hardware platforms. Most
"CI is red" reports are one of: wrong platform routing, a stale plan, a missing
runner/image, a required group that produced no report, or an advisory upload
being mistaken for a failure. Diagnose in that order.

## Pipeline at a glance

```
ci.yml
  lint ──► plan ──► platform-pipeline (matrix, one job per platform)
                        └─► _platform_pipeline.yml ─► all_tests_common.yml
                               checkout_and_config → preflight → build_test_artifact
                               → test groups (parallel / wave0 / wave1 / serial)
                               → coverage → platform_gate
            └─► ci-complete (flagcicd-gate, runs unless cancelled)
```

Entry points:

| Workflow | Trigger | Stage / level |
|---|---|---|
| `ci.yml` | push to `main`; PR to `main`/`devops` (`labeled`/`unlabeled` included); dispatch | `pr` or `branch`, auto level |
| `ci-requested.yml` | PR comment `/ci run <platforms\|all>` or `/test ...`; dispatch (`platforms`, `test_level`, `checkout_ref`) | `manual`, default L2 |
| `daily.yml` | cron `17 18 * * *` (02:17 Asia/Shanghai); dispatch | `nightly`, L3, all platforms |
| `release-acceptance.yml` | dispatch with `checkout_ref` only | `release`, L4 |

`trigger-ci.yml`, `blossom-ci.yml` and `qa-l0-*` / `qa-l1-*` / `qa-l3-*` are
disabled (their triggers use `__disabled_do_not_remove__`). Do not debug them.
`qa-format.yml` still runs on PRs to `main`.

## Triage: first three commands

```sh
gh run list -L 10 --json databaseId,name,status,conclusion,headBranch,event \
  --jq '.[] | "\(.databaseId) \(.conclusion // .status) \(.name) \(.headBranch)"'
gh run view <run-id> --log-failed          # only the failing steps
gh pr checks <pr-number>                   # per-check state
```

Then read the `plan` job output before touching any test job. It names the
chosen platforms, level and the reason (the planner writes reasons to
`GITHUB_STEP_SUMMARY` and stderr).

## Platform matrix

The single source of truth is `.github/configs/<platform>.yml`; template in
`template.yml`. Key fields: `hardware_name`, `ci_image`,
`container_pull_policy` (default `never`), `runner_labels`, `setup_script`,
`test_artifact`, `device_types`, `coverage`, `nproc_per_node`,
`unit_test_matrix` / `integration_test_matrix` / `compat_test_matrix`,
`paths_watch`.

| Platform | Runner label | Coverage required |
|---|---|---|
| cuda | `nv-8g-cicd-te` | no (has `test_artifact.enabled`) |
| ascend | `flagcicd-910c` | **yes** |
| enflame | `sy-8g-cicd-te` | no |
| hygon | `hg-8g-cicd-te` | no |
| kunlun | `kl-8g-cicd-te` | no |
| metax | `mx-8g-cicd-te` | no |
| musa | `mt-8g-cicd-te` | no |
| ppu | `flagcicd-810e` | no |

Images live on `harbor.baai.ac.cn`. Default `container_pull_policy: never`
means the runner must already hold the image.

## Routing rules (why the wrong platforms ran)

`generate_matrix.py::select_platforms` is imported by the planner:

- Non-PR event → **all** platforms.
- Label `platform:<name>` overrides changed-file routing; `platform:all` → all.
- No changed-file information, diff failure, or any platform with empty
  `paths_watch` → **all** platforms.
- Only documentation (`docs/`, `*.md`, `*.rst`) → **empty** matrix, `lint_only`.
- Any non-documentation path claimed by no platform → **all** platforms.
- Otherwise: the union of platforms whose `paths_watch` matches the diff.
  Entries ending in `/` are prefixes, everything else is an exact path.
  Renames count both old and new paths (`--no-renames`).

Consequence: a new top-level directory that no platform watches silently runs
the full matrix. Add it to the relevant `paths_watch` (or to
`template.yml`) in the same PR.

## Test levels

Defined in `.flagcicd/project.yaml` (`levels`, `stages`, `risk_rules`).
Stage defaults: PR=L1 (affected only), branch=L2 (all platforms),
nightly=L3 (fixed), manual=L2 (requested), release=L4. `level <= test_level`
directory groups run. Risk rules raise the level by changed path (`.github/**`
and build files → L2, native `csrc`/CUDA → L3); unmatched files → L2. Label
`level:L0`…`L4` can only raise, never lower, the computed level.

Groups marked `blocked` in `project.yaml` make L4 (`release-acceptance`) fail by
design until they are implemented — do not "fix" that by relaxing the gate.

## Failure playbook

| Symptom | Likely cause | Check / fix |
|---|---|---|
| Job queued forever | runner label typo or runner offline | compare `runner_labels` with the platform; `gh api /repos/{owner}/{repo}/actions/runners` |
| `Platform not in plan` | platform not selected by planner | inspect `plan` output; check config stem, label, `paths_watch` |
| Config/HEAD mismatch in slice/gate | commit pushed after planning | re-run the whole workflow; plan pins `source_sha` and config sha256 |
| Image pull error | wrong harbor tag, or `pull_policy: never` with no local image | `docker images` on the runner; fix `ci_image` |
| `yq` download step fails | runner cannot reach GitHub releases | pre-provision `yq` or pin a mirror |
| Gate: required group missing report | group crashed before upload, or `executed=false` | inspect `group-report` artifacts (`if-no-files-found: error`) |
| Ascend coverage failure | `coverage.required: true` and no fragments | run the coverage job; do not mark it optional |
| Feishu notification missing | `notify-feishu.yml` listens to `workflow_run` of CI/Daily | check `FEISHU_*` secrets; script exits 0 when unset |
| FlagOps upload error | advisory only (`fail_on_error: false`) | harmless unless job status differs |

Test-group execution retries at most twice (`TE_TEST_MAX_RETRIES=2`), 5s apart;
signal exits (`<0` or `>=128`) are never retried. Group timeouts come from the
matrix (default 90m; cuda/ppu unit 240m).

Do not paper over gate failures with `continue-on-error` or by adding
`always()`. The gate intentionally fails on missing required groups.

## Local validation (no hardware needed)

```sh
# Same self-test the lint job runs
python -m unittest discover -s .github/scripts/ci -p 'test_*.py'

# Generate a plan without pushing
python .github/scripts/ci/flagcicd_planner.py --stage nightly --output /tmp/plan.json
python .github/scripts/ci/flagcicd_planner.py --stage pr --base-ref main --labels-from-event

# Exercise routing directly
python .github/scripts/ci/generate_matrix.py --event-name pull_request \
  --base-ref main --changed-files "$(printf 'transformer_engine/pytorch/foo.py\nREADME.md')"

# Formatting
pre-commit run --show-diff-on-failure --from-ref origin/main --to-ref HEAD
bash qa/format.sh
```

Local runs cannot validate hardware compatibility, nested matrix behavior, or
runner availability — say so instead of claiming a fix is verified.

## File map

- Entry workflows: `.github/workflows/ci.yml`, `ci-requested.yml`, `daily.yml`,
  `release-acceptance.yml`
- Reusable: `_platform_pipeline.yml`, `all_tests_common.yml`,
  `test_groups_common.yml`, `unit_tests_common.yml`,
  `integration_tests_common.yml`
- Actions: `.github/actions/flagcicd-plan`, `.github/actions/flagcicd-gate`
- Planning/gating: `.github/scripts/ci/flagcicd_planner.py`,
  `flagcicd_gate.py`, `generate_matrix.py`, `ci_requests.py`
- Config: `.flagcicd/project.yaml`, `.github/configs/*.yml`
- Notifications: `.github/workflows/notify-feishu.yml`,
  `.github/scripts/notify_feishu.py`
