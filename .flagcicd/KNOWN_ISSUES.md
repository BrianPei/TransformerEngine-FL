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
**Status**: ❌ NOT FIXED - Requires architecture change
**Files**:
- `.flagcicd/project.yaml` (risk_rules defined)
- `.github/scripts/ci/flagcicd_planner.py` (resolve_level, build_test_groups)
**Issue**:
- `resolve_level()` only returns stage default, doesn't match risk_rules patterns
- Test groups' `paths_watch` not used for filtering
- `.github/**` changes get L1 instead of required L2
- Integration/native changes don't escalate test level
**Impact**: High-risk changes (CI changes, native extensions) tested at insufficient level
**Remediation Plan**:
1. Implement `match_risk_rules()` with fnmatch glob matching
2. Add `paths_watch` filtering in `build_test_groups()`
3. Return `(test_level, risk_level)` tuple from `resolve_level()`
4. Add label-based escalation (`level:L2`, etc.)
5. Update all call sites and tests
**Estimated Effort**: 2-3 hours
**Blocker**: Changes resolve_level signature, affects all downstream code

### P1-4: Platform/Global Gate Validation Incomplete
**Status**: ❌ NOT FIXED - Security/correctness issue
**Files**: `.github/scripts/ci/flagcicd_gate.py` (cmd_platform, cmd_gate)
**Issue**:
- `cmd_platform()` records errors but doesn't fail on them
- Missing/duplicate/unexpected reports don't cause failure
- `cmd_gate()` doesn't validate plan_hash, source_sha, repository, stage
- Empty platform reports with `status=passed` accepted
**Impact**: Stale artifacts, wrong-plan execution, or corrupted reports pass CI
**Remediation Plan**:
1. `cmd_platform()`: Return non-zero if `errors` list non-empty
2. Add schema validation for all report types
3. Validate plan/source/stage/repository consistency
4. Validate required groups have `executed=true`
5. Fail closed on: missing reports, hash mismatches, schema errors
6. Add unit tests for gate validation logic
**Estimated Effort**: 3-4 hours
**Blocker**: Needs comprehensive test coverage before deploying stricter validation

### P1-5: L0 Preflight Not Executed
**Status**: ❌ NOT FIXED - Missing job
**Files**:
- `.github/scripts/ci/flagcicd_planner.py` (adds preflight to plan)
- `.github/workflows/all_tests_common.yml` (no preflight job)
**Issue**: Planner generates required preflight groups, but workflow has no corresponding job
**Impact**: Required L0 checks silently skipped; if gate fixed, all platforms fail
**Remediation Plan**:
1. Add `preflight` job to `all_tests_common.yml` before `build_test_artifact`
2. Use `needs.checkout_and_config.outputs.preflight_selected`
3. Run quick validation (lint subset, fast unit smoke tests)
4. Or: Remove preflight from required plan if not implementing
**Estimated Effort**: 1-2 hours

### P1-6: Coverage Collection Broken
**Status**: ❌ NOT FIXED - Data loss
**Files**:
- `.github/workflows/test_groups_common.yml` (no coverage env setup)
- `tests/test_utils/run_ci_test_group.py` (doesn't generate .coverage)
- `.github/workflows/all_tests_common.yml` (coverage job assumptions)
**Issue**:
- No `COVERAGE_ENABLED`, `PYTEST_ADDOPTS`, `coverage run` setup
- `run_ci_test_group.py` doesn't instrument tests
- Coverage job exits 0 when no fragments found
- Ascend `coverage: required: true` not enforced
**Impact**: Coverage regression checks ineffective, required coverage silently skipped
**Remediation Plan**:
1. Restore coverage environment variables in test group execution
2. Pass `coverage_sources`, `coverage_include`, `coverage_omit` from group config
3. Modify `run_ci_test_group.py` to run under `coverage run` when enabled
4. Make coverage job fail when `required: true` and no fragments
5. Validate coverage job result in platform gate
**Estimated Effort**: 3-4 hours
**Blocker**: Needs coordination with test execution changes

### P1-7: Legacy Workflows Broken
**Status**: ❌ NOT FIXED - Backward compatibility break
**Files**:
- `all_tests_{cuda,ascend,enflame,hygon,kunlun,metax,musa,ppu}.yml` (8 files)
- `.github/workflows/ci-requested.yml`
- `.github/workflows/flagcicd-entry.yml`
**Issue**:
- `all_tests_common.yml` requires `plan` input (new), but legacy callers don't provide it
- `flagcicd-entry.yml` calls deleted `flagcicd_plan.py`, `aggregate_flagcicd_reports.py`
- Legacy workflows pass deleted inputs: `run_unit_tests`, `run_integration_tests`, `test_level`
**Impact**: Manual platform workflows, `/ci run` command, legacy entry points fail immediately
**Remediation Plan**:
Option A (Clean Break):
1. Update all 8 `all_tests_*.yml` to use new `ci.yml` pattern (plan -> pipeline)
2. Delete `ci-requested.yml` and `flagcicd-entry.yml`
3. Document migration in `.flagcicd/README.md`

Option B (Compatibility Layer):
1. Make `plan` input optional with default value
2. Create `plan` from legacy inputs when not provided
3. Add deprecation warnings
4. Set removal date

**Estimated Effort**: 2-3 hours (Option A), 4-5 hours (Option B)
**Recommendation**: Option A - clean break preferred for v1 launch

### P1-8: Blocked Groups Included in Execution
**Status**: ❌ NOT FIXED - Incorrect planning
**Files**:
- `.flagcicd/project.yaml` (blocked groups defined)
- `.github/scripts/ci/flagcicd_planner.py` (build_test_groups)
**Issue**:
- `pytorch_benchmark` marked `blocked: "no baseline"` but included in nightly plan
- Group has `exec: null`, causes execution failure
- `run_fa_versions_tests.sh` referenced but doesn't exist
- Planner only checks `blocked` when `matrix` exists
**Impact**: Nightly fails on blocked benchmark group
**Remediation Plan**:
1. Check `blocked` field for all groups, not just matrix groups
2. Add blocked groups to plan with `status: blocked`
3. Exclude blocked groups from execution matrices
4. Gate validates blocked groups stayed blocked (not executed)
5. Remove references to non-existent scripts or create stubs
6. Document blocked group lifecycle in `.flagcicd/README.md`
**Estimated Effort**: 2 hours

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
