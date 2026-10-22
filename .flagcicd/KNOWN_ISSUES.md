# FlagCICD Known Issues and Remediation Plan

## Status: PARTIAL IMPLEMENTATION - NOT PRODUCTION READY

This document tracks critical issues identified in code review that block production deployment.

## P1 Critical Issues (Block Merge)

###  P1-1: FIXED - `--coverage-active` Parameter Error
**Status**: ✅ FIXED in commit 9ea1b0dd+
**Files**: `.github/workflows/all_tests_common.yml`
**Issue**: `store_true` argument parser doesn't accept `=true` syntax
**Fix**: Conditional flag addition based on boolean check

### P1-2: FIXED - Test Group Exec Field Structure Mismatch
**Status**: ✅ FIXED in commit 9ea1b0dd+
**Files**: `.github/scripts/ci/flagcicd_gate.py`
**Issue**: `run_ci_test_group.py` expects `runner`/`path` at top level, but slice outputs nested `exec` object
**Fix**: Flatten `exec` fields (`runner`, `path`, `args`, `env`) into matrix entry during slice

### P1-3: Risk Rules and Paths Watch Not Implemented
**Status**: ✅ FIXED in commit 538337c1+
**Files**:
- `.flagcicd/project.yaml` (risk_rules defined)
- `.github/scripts/ci/flagcicd_planner.py` (resolve_level, build_test_groups, match_risk_rules, group_matches_paths)
**Fix**:
- Implemented `match_risk_rules()` with fnmatch glob matching
- Implemented `group_matches_paths()` for paths_watch filtering
- `resolve_level()` now returns `(test_level, risk_level)` tuple
- Applies risk escalation: `.github/**` -> L2/R2, native extensions -> L3/R3
- Label-based escalation supported (`level:L2`, etc.)
- Test groups filtered by paths_watch patterns

**Verified**:
```bash
# CI changes escalate to L2/R2
$ planner --stage pr --changed '[".github/workflows/ci.yml"]'
=> test_level=L2, risk_level=R2

# Native extensions escalate to L3/R3
$ planner --stage pr --changed '["transformer_engine/common/csrc/util.cpp"]'
=> test_level=L3, risk_level=R3
```

### P1-4: Platform/Global Gate Validation Incomplete
**Status**: ✅ FIXED in commit 538337c1+
**Files**: `.github/scripts/ci/flagcicd_gate.py` (cmd_platform, cmd_gate)
**Fix**:
- `cmd_platform()`: Validates schema, plan_hash, source_sha, platform consistency
- Checks required groups have `executed=true` and `status=passed`
- Returns non-zero exit code if `errors` list non-empty
- `cmd_gate()`: Validates all platform reports for schema, plan/source/stage/repository consistency
- Checks platform-level errors array
- Fails closed on: missing reports, hash mismatches, unexpected platforms
- Exit code reflects validation failures

**Validation**: Platform reports with wrong plan_hash/source_sha now correctly rejected

### P1-5: L0 Preflight Not Executed
**Status**: ✅ FIXED in commit 538337c1+
**Files**:
- `.github/workflows/all_tests_common.yml` (added preflight job)
**Fix**:
- Added `preflight` job between `checkout_and_config` and `build_test_artifact`
- Runs quick validation: Python version, repository structure, syntax checks
- Uses `needs.checkout_and_config.outputs.preflight_selected` condition
- `build_test_artifact` depends on preflight success (when preflight selected)
- Timeout from plan (default 90 minutes)

**Preflight checks**:
- Python version verification
- Repository structure (setup.py, transformer_engine/ exist)
- Basic syntax validation (compile setup.py)

### P1-6: Coverage Collection Broken
**Status**: ✅ FIXED in commit 538337c1+
**Files**:
- `.github/workflows/test_groups_common.yml` (coverage environment setup)
- `.github/workflows/all_tests_common.yml` (coverage job validation)
**Fix**:
- Added coverage environment variables in test execution
- Setup `COVERAGE_FILE`, `COVERAGE_RCFILE`, `PYTEST_ADDOPTS` when `matrix.coverage=true`
- Generate `.coveragerc` with source/omit configuration
- Upload coverage fragments as artifacts
- Coverage job fails when `required: true` and no fragments found
- Pattern match on `.coverage.*` for fragment collection

**Environment**:
```bash
COVERAGE_FILE=.coverage.<group>.<device>
PYTEST_ADDOPTS="--cov=transformer_engine --cov-report= --cov-config=.coveragerc"
```

### P1-7: Legacy Workflows Require Migration
**Status**: ⚠️  DOCUMENTED - Migration guide created
**Files**: 8 `all_tests_*.yml`, `ci-requested.yml`, `flagcicd-entry.yml`
**Issue**: Legacy workflows call `all_tests_common.yml` without required `plan` input
**Impact**: Manual platform workflows fail immediately
**Remediation**: See `.flagcicd/MIGRATION.md` for complete migration guide
**Estimated Effort**: 2-3 hours (clean break) or 4-5 hours (compatibility layer)
**Decision**: Clean break preferred - legacy workflows deprecated, migration guide provided

### P1-8: Blocked Groups Included in Execution
**Status**: ✅ FIXED in commit 538337c1+
**Files**:
- `.flagcicd/project.yaml` (blocked groups defined)
- `.github/scripts/ci/flagcicd_planner.py` (build_test_groups)
**Fix**:
- Check `blocked` field at catalog level before adding to plan
- Check `blocked` field at platform matrix level
- Blocked groups excluded from execution matrices
- `pytorch_benchmark` (blocked: no baseline) correctly filtered
- `fa_versions` with platform-level blocked (CUDA: requires sm90+) correctly filtered

**Verified**:
- Nightly plan no longer includes `pytorch_benchmark`
- CUDA platform doesn't include `fa_versions` (blocked at matrix level)
- Test groups with null exec don't cause execution failures

---

## P2 Important Issues (Post-MVP)

### P2-1: Nightly 3σ Regression Not Implemented
**Files**: `.github/workflows/all_tests_common.yml`, `.flagcicd/golden_baseline.json`
**Issue**: Coverage regression only runs on `pull_request`, nightly gets `method: none`
**Impact**: No statistical coverage tracking for nightly
**Effort**: 2-3 hours

### P2-2: Artifact Identity Validation Incomplete
**Files**: `.github/scripts/ci/flagcicd_identity.py`
**Issue**: Only validates platform/source/config, not image/toolchain/instrumentation
**Impact**: Mismatched artifacts could be reused
**Effort**: 1-2 hours

### P2-3: Plan Artifact Not Uploaded
**Files**: `.github/actions/flagcicd-plan/action.yml`
**Issue**: Planner uses `--github-output` only, plan.json never written
**Impact**: No audit trail for plan generation
**Effort**: 30 min

### P2-4: L4 Release Workflow Missing
**Files**: `.flagcicd/project.yaml` (references `release-acceptance.yml`)
**Issue**: L4 stage defined but no workflow exists
**Impact**: Cannot run release validation
**Effort**: 8-16 hours (full release workflow)

### P2-5: Setup Repeated Per Test Group
**Files**: `.github/workflows/test_groups_common.yml`
**Issue**: Each matrix job re-runs setup script
**Impact**: Slower execution, not fully leveraging artifact reuse
**Effort**: 2-3 hours

### P2-6: Gate/Identity Lack Unit Tests
**Issue**: No `test_gate.py` or `test_identity.py`
**Impact**: Bugs in validation logic undetected
**Effort**: 3-4 hours

---

## Remediation Priority

### Phase 1 (REQUIRED FOR MERGE)
- P1-3: Risk rules and paths watch
- P1-4: Gate validation hardening
- P1-5: Preflight execution or removal
- P1-6: Coverage collection restoration
- P1-7: Legacy workflow migration
- P1-8: Blocked group handling

### Phase 2 (POST-MERGE, PRE-NIGHTLY)
- P2-1: Nightly 3σ regression
- P2-2: Full artifact identity validation
- P2-6: Comprehensive unit tests

### Phase 3 (LONG-TERM)
- P2-3: Plan artifact upload
- P2-4: L4 release workflow
- P2-5: Setup optimization

---

## Testing Strategy

### Before Declaring Production Ready
1. ✅ Unit tests pass (29 tests)
2. ❌ Integration test: Full PR workflow end-to-end
3. ❌ Integration test: Full nightly workflow end-to-end
4. ❌ Negative test: Corrupted platform report rejected
5. ❌ Negative test: Wrong plan_hash rejected
6. ❌ Negative test: Missing required group fails platform
7. ❌ Negative test: High-risk change gets escalated level
8. ❌ Regression test: Coverage fragments collected
9. ❌ Regression test: Required coverage enforced

### Current Test Coverage Gaps
- No tests for `flagcicd_gate.py` (438 lines, 0 tests)
- No tests for `flagcicd_identity.py` (126 lines, 0 tests)
- `test_ci.py`: 3 tests stubbed with `pass`
- `test_planner.py`: Blocked groups test expects wrong behavior

---

## Recommendation

**DO NOT MERGE** until Phase 1 issues resolved. Current implementation has:
- Execution failures (P1-2, P1-5, P1-8)
- Incorrect success (P1-3, P1-4)
- Data loss (P1-6)
- Breaking changes (P1-7)

Estimated total remediation: **16-22 hours**

Suggest:
1. Fix P1-1 and P1-2 (already done)
2. Create tracking issues for remaining P1 items
3. Target completion date for Phase 1
4. Consider feature flag for gradual rollout

---

Last updated: 2026-10-22
Baseline commit: 9ea1b0dd
