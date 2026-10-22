# FlagCICD v1 Migration Guide

## Overview

FlagCICD v1 introduces plan-driven execution. Legacy per-platform workflows must migrate to the new pattern.

## Legacy Workflows (DEPRECATED)

The following workflows are deprecated and will be removed:
- `.github/workflows/all_tests_cuda.yml`
- `.github/workflows/all_tests_ascend.yml`
- `.github/workflows/all_tests_enflame.yml`
- `.github/workflows/all_tests_hygon.yml`
- `.github/workflows/all_tests_kunlun.yml`
- `.github/workflows/all_tests_metax.yml`
- `.github/workflows/all_tests_musa.yml`
- `.github/workflows/all_tests_ppu.yml`
- `.github/workflows/ci-requested.yml`
- `.github/workflows/flagcicd-entry.yml`

## New Entry Points

### PR and Branch Push
Use `.github/workflows/ci.yml`:
- Triggered automatically on `pull_request` and `push` to `main`
- Generates plan based on changed files
- Runs affected platforms only (PR) or all platforms (branch)

### Nightly
Use `.github/workflows/nightly.yml`:
- Triggered on schedule or manual dispatch
- Runs all platforms at L3 level
- Includes FA version matrix and benchmarks (when unblocked)

### Manual Platform Run
Use workflow_dispatch on `.github/workflows/ci.yml`:
```yaml
inputs:
  platforms: '["cuda", "ascend"]'  # JSON array
  test_level: 'L2'                 # Optional, default: L1
```

Or use GitHub CLI:
```bash
gh workflow run ci.yml \
  -f platforms='["cuda"]' \
  -f test_level=L2
```

## Migration Steps

### For Platform-Specific Manual Runs

**OLD** (all_tests_cuda.yml):
```yaml
name: CUDA Tests
on: [workflow_dispatch, push]
jobs:
  tests:
    uses: ./.github/workflows/all_tests_common.yml
    with:
      platform: cuda
      checkout_ref: ${{ github.sha }}
      run_unit_tests: true
      run_integration_tests: true
```

**NEW** (use ci.yml with workflow_dispatch):
```bash
# Via GitHub UI: Actions -> CI -> Run workflow -> Select branch -> Fill inputs
# Via CLI:
gh workflow run ci.yml \
  -f platforms='["cuda"]' \
  -f test_level=L2
```

### For Scheduled/Nightly Runs

**OLD** (flagcicd-entry.yml):
```yaml
on:
  schedule:
    - cron: '0 0 * * *'
```

**NEW**: Already implemented in `nightly.yml`

### For `/ci run` Comment Commands

**Status**: Command detection works (PR comment parsing), but requires workflow_dispatch integration

**Implementation**: The new `ci.yml` supports plan generation from labels and manual platform selection. To restore `/ci run <platform>` functionality:

1. Parse PR comment in workflow
2. Extract platform list
3. Call plan generation with `--platforms` override
4. Execute selected platforms

**Example** (to be added to ci.yml):
```yaml
on:
  issue_comment:
    types: [created]

jobs:
  comment_handler:
    if: |
      github.event.issue.pull_request &&
      startsWith(github.event.comment.body, '/ci run')
    runs-on: ubuntu-latest
    steps:
      - name: Parse command
        id: parse
        run: |
          # Extract platforms from "/ci run cuda ascend"
          PLATFORMS=$(echo "${{ github.event.comment.body }}" | sed 's|/ci run||' | tr ' ' '\n' | grep -v '^$' | jq -R -s -c 'split("\n")[:-1]')
          echo "platforms=$PLATFORMS" >> "$GITHUB_OUTPUT"

      - name: Trigger CI
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.actions.createWorkflowDispatch({
              owner: context.repo.owner,
              repo: context.repo.repo,
              workflow_id: 'ci.yml',
              ref: context.payload.issue.pull_request.head.ref,
              inputs: {
                platforms: '${{ steps.parse.outputs.platforms }}',
                test_level: 'auto'
              }
            });
```

## Breaking Changes

### Removed Inputs (all_tests_common.yml)

- `run_unit_tests` -> Plan determines groups based on level
- `run_integration_tests` -> Plan determines groups based on level
- `test_level` -> Passed in plan, not workflow input
- `build_artifact` -> Determined by platform config (`test_artifact.enabled`)

### New Required Input

- `plan` (string): JSON plan from flagcicd-plan action

### Removed Scripts

- `.github/scripts/flagcicd_plan.py` -> `.github/scripts/ci/flagcicd_planner.py`
- `.github/scripts/aggregate_flagcicd_reports.py` -> `.github/scripts/ci/flagcicd_gate.py`

## Testing Migration

Before removing legacy workflows:

1. ✅ Verify `ci.yml` generates plan correctly
2. ✅ Verify platform-pipeline calls all_tests_common with plan
3. ✅ Verify test groups execute and generate reports
4. ✅ Verify gate validation works
5. ❌ Test workflow_dispatch with custom platforms
6. ❌ Test `/ci run` comment command (requires implementation)

## Rollout Plan

### Phase 1 (Current)
- Keep legacy workflows for backward compatibility
- Mark as deprecated in documentation
- New PRs use `ci.yml` by default

### Phase 2 (After validation)
- Add `deprecated: true` to legacy workflow metadata
- Add banner warning to workflow logs
- Set removal date (e.g., 30 days notice)

### Phase 3 (Removal)
- Delete legacy workflow files
- Update branch protection rules if needed
- Remove from documentation

## FAQ

### Q: How do I run tests on a specific platform?
A: Use `workflow_dispatch` on `ci.yml` with `platforms` input

### Q: How do I run only unit tests?
A: Set `test_level: L1` (unit only) or `L2` (unit + integration)

### Q: Can I still use platform labels?
A: Yes, `platform:cuda` label routes to CUDA platform automatically

### Q: What happened to the test artifact caching?
A: Still works - configured in platform configs under `test_artifact.enabled`

### Q: How do I see the execution plan?
A: Check the `plan` job output in the workflow run

## Support

For issues or questions:
1. Check `.flagcicd/KNOWN_ISSUES.md` for known limitations
2. Check `.flagcicd/README.md` for architecture documentation
3. File GitHub issue with `cicd` label
