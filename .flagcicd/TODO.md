# FlagCICD Remaining Tasks

## Current Status

✅ **Phase 1 Complete**: All P1 blocking issues resolved (commit 251ffb0b)
- Risk rules and paths_watch working
- Gate validation hardened
- Preflight executing
- Coverage collecting
- Blocked groups filtered

## Nightly Full Coverage Issue

### Issue: L3 Test Groups are Blocked

**Current Behavior**:
Nightly runs at L3 level but only executes L0-L2 groups:
- ✓ preflight (L0) - 8 platforms
- ✓ pytorch_unittest (L1) - 8 platforms  
- ✓ pytorch_mcore_integration (L2) - 8 platforms
- ✗ fa_versions (L3) - BLOCKED on all platforms
- ✗ pytorch_benchmark (L3) - BLOCKED on all platforms

**Total**: 24 test groups (should be 40+ if L3 unblocked)

**Root Causes**:
1. `fa_versions` blocked:
   - CUDA: "requires sm90+ (Hopper); runners are A100 (sm80)"
   - Other platforms: no `compat_test_matrix` defined
   - Referenced script: `tests/plugin/backend/cuda/run_fa_versions_tests.sh` (doesn't exist)

2. `pytorch_benchmark` blocked:
   - All platforms: "no performance baseline or benchmark launcher is defined"
   - No `benchmark_test_matrix` in any platform config
   - No benchmark runner scripts

**Impact**: Nightly is effectively running L2 tests, not L3

### Solutions

#### Option A: Create L3 Test Infrastructure (HIGH EFFORT, ~16-24 hours)

**For fa_versions**:
1. Create `tests/plugin/backend/cuda/run_fa_versions_tests.sh`
2. Define FA version matrix (2.5.7, 2.5.8, 2.5.9, 2.6.1, 2.6.3)
3. Test installation/compatibility per version
4. For Hopper requirement:
   - Either: Get Hopper runner access
   - Or: Relax to A100-compatible FA tests
5. Add `compat_test_matrix` to other platforms

**For pytorch_benchmark**:
1. Create benchmark runner script
2. Define performance baseline (golden_baseline.json extension)
3. Add `benchmark_test_matrix` to platform configs
4. Implement regression detection (3-sigma or percentage)

**Estimated Effort**: 16-24 hours

#### Option B: Add Substitute L3 Tests (MEDIUM EFFORT, ~6-8 hours)

Create new L3 groups that don't require blocked infrastructure:

1. **pytorch_extended_integration** (L3):
   - Longer-running integration tests
   - Multi-GPU scenarios
   - Stress tests
   - Matrix: existing integration_test_matrix with extended entries

2. **numerical_accuracy** (L3):
   - Reference implementation comparison
   - Dtype conversion accuracy
   - Numerical stability tests
   - New matrix in platform configs

3. **performance_smoke** (L3):
   - Basic throughput checks (not full benchmark)
   - Regression detection against previous run
   - No golden baseline required
   - Simple threshold: ±20% of previous

**Estimated Effort**: 6-8 hours

#### Option C: Redefine Level Semantics (LOW EFFORT, ~1 hour)

Accept current state and adjust documentation:

1. Update level definitions:
   - L2: unit + integration (current "nightly")
   - L3: FA versions + benchmarks (blocked, future)
   - Document that nightly runs L2, not L3

2. Change nightly default_level to L2
3. Update KNOWN_ISSUES.md and README.md
4. Set target date for unblocking L3

**Estimated Effort**: 1 hour

**Recommendation**: Option C for immediate unblock, then pursue Option B in parallel with Option A

---

## P2 Enhancements (Post-MVP)

### P2-1: Nightly 3σ Coverage Regression ⚠️ IMPORTANT

**Status**: Not implemented
**Impact**: No statistical coverage tracking for nightly
**Files**: `.github/workflows/all_tests_common.yml`, `.flagcicd/golden_baseline.json`

**Issue**:
- Coverage regression only runs on `github.event_name == 'pull_request'`
- Nightly plan outputs `coverage_policy: {method: "none"}`
- `golden_baseline.json` defines 3σ policy but not implemented

**Tasks**:
1. Change coverage job condition to include nightly
2. Implement historical window loading (last 14 nightly runs)
3. Calculate mean and stddev
4. Check if current < mean - 3*stddev
5. Store result in nightly coverage artifact
6. Update `golden_baseline.json` with initial baseline

**Estimated Effort**: 2-3 hours

---

### P2-2: Artifact Identity Full Validation

**Status**: Partial validation
**Impact**: Mismatched artifacts could be reused
**Files**: `.github/scripts/ci/flagcicd_identity.py`

**Current Validation**:
- ✓ platform
- ✓ source_sha
- ✓ config_hash
- ✗ schema_version
- ✗ image digest (only ref)
- ✗ test_level
- ✗ coverage_active
- ✗ toolchain versions
- ✗ wheel checksums

**Tasks**:
1. Add schema version check
2. Validate image digest (not just ref)
3. Check test_level match
4. Verify coverage instrumentation flag
5. Store and verify wheel SHA256
6. Optional: capture and verify toolchain (gcc, nvcc, python versions)

**Estimated Effort**: 1-2 hours

---

### P2-3: Plan Artifact Upload

**Status**: Plan not persisted
**Impact**: No audit trail for plan generation
**Files**: `.github/actions/flagcicd-plan/action.yml`

**Issue**:
Planner uses `--github-output` only, `plan.json` never written to disk

**Tasks**:
1. Add `--output plan.json` to planner invocation
2. Upload `plan.json` as artifact
3. Add plan download step to gate job
4. Validate downloaded plan matches needs.plan.outputs.plan

**Estimated Effort**: 30 minutes

---

### P2-4: L4 Release Workflow

**Status**: Not implemented
**Impact**: Cannot run release validation
**Files**: `.flagcicd/project.yaml` (references `.github/workflows/release-acceptance.yml`)

**Required Components**:
1. Unblock 5 L4 groups (all currently blocked):
   - cross_repo_integration
   - model_validation
   - release_wheel_validation
   - backward_compatibility
   - golden_baseline_acceptance

2. Create infrastructure:
   - Megatron-LM-FL/FlagScale pinned versions
   - Model training jobs (e.g., GPT-3 convergence check)
   - Release wheel builder
   - Clean environment tester
   - Previous version compatibility matrix
   - Numerical golden values

3. Create `release-acceptance.yml` workflow

**Estimated Effort**: 8-16 hours (MVP), 40+ hours (full implementation)

**Recommendation**: Defer to post-v1, not required for initial deployment

---

### P2-5: Setup Optimization

**Status**: Setup repeated per test group
**Impact**: Slower execution
**Files**: `.github/workflows/test_groups_common.yml`

**Current Behavior**:
Each matrix job runs:
```yaml
- name: Setup
  run: bash "$GITHUB_WORKSPACE/${{ inputs.setup_script }}"
```

**Issue**:
- Setup script runs N times (once per group)
- Dependency installation repeated
- Environment initialization repeated
- Only artifact download is reused

**Tasks**:
1. Move setup to a dedicated job after checkout_and_config
2. Capture installed environment as artifact
3. Test groups download and activate environment
4. Or: use pre-built images with dependencies

**Estimated Effort**: 2-3 hours

**Blocker**: Platform-specific setup scripts may have side effects

---

### P2-6: Comprehensive Unit Tests for Gate and Identity

**Status**: No dedicated tests
**Impact**: Validation logic bugs may go undetected
**Files**: Missing `test_gate.py`, `test_identity.py`

**Current Coverage**:
- `flagcicd_gate.py` (442 lines): 0 tests
- `flagcicd_identity.py` (126 lines): 0 tests
- `flagcicd_contract.py` (86 lines): 0 tests

**Test Scenarios Needed**:

**test_gate.py**:
- cmd_slice: matrix generation, preflight selection
- cmd_group_report: status determination from steps + junit
- cmd_platform: report aggregation, error detection, required group validation
- cmd_gate: multi-platform validation, hash checking

**test_identity.py**:
- write: artifact identity generation
- verify: identity validation, mismatch detection

**test_contract.py**:
- load_plan: schema validation
- canonical_json: deterministic ordering
- compute_plan_hash: consistency

**Estimated Effort**: 3-4 hours

---

### P2-7: Workflow Dispatch Improvements

**Status**: Basic workflow_dispatch exists
**Impact**: Manual runs less convenient
**Files**: `.github/workflows/ci.yml`

**Current Limitations**:
- Can specify platforms and level
- But no PR context (must select branch manually)
- No `/ci run` comment command integration

**Tasks**:
1. Add issue_comment trigger to ci.yml
2. Parse `/ci run <platform> <platform>` syntax
3. Extract PR head ref
4. Trigger workflow_dispatch with parsed platforms
5. Post comment with workflow run link

**Estimated Effort**: 2 hours

---

### P2-8: Coverage Reporting Enhancements

**Status**: Basic coverage collection works
**Impact**: Coverage data underutilized

**Missing**:
- HTML coverage report generation
- Coverage badge in README
- Per-module coverage breakdown
- Coverage trend visualization
- Diff coverage (only changed lines)

**Tasks**:
1. Generate HTML report in coverage job
2. Upload as GitHub Pages artifact
3. Add badge to README.md
4. Use coverage comment action for PR feedback
5. Implement diff coverage with coverage.py

**Estimated Effort**: 2-3 hours

---

## Testing Requirements Before Production

### Integration Tests Needed

1. ❌ **Full PR Workflow End-to-End**
   - Fork repo, create PR with CI changes
   - Verify level escalates to L2/R2
   - Verify gate passes

2. ❌ **Full Nightly Workflow End-to-End**
   - Trigger nightly manually
   - Verify 8 platforms execute
   - Verify 24 groups complete
   - Verify platform reports generated
   - Verify gate passes

3. ❌ **Negative Test: Corrupted Platform Report**
   - Inject wrong plan_hash
   - Verify gate rejects

4. ❌ **Negative Test: Missing Required Group**
   - Simulate group failure
   - Verify platform report fails
   - Verify gate fails

5. ❌ **Negative Test: Wrong Source SHA**
   - Use stale artifact with old SHA
   - Verify identity verification fails
   - Verify tests don't execute

6. ❌ **Risk Escalation Test**
   - PR with native extension change
   - Verify level escalates to L3/R3

7. ❌ **Coverage Collection Test**
   - Verify fragments uploaded
   - Verify aggregation succeeds
   - Verify regression check runs

8. ❌ **Preflight Test**
   - Verify preflight job executes before build
   - Verify build depends on preflight success

---

## Priority Roadmap

### Immediate (Before Merge to Main)
1. ✅ All P1 issues (COMPLETE)
2. 🔄 Choose Nightly L3 strategy (Option C recommended)
3. ❌ Run integration tests 1-8 above
4. ❌ Fix any issues found in integration tests

### Short-term (Next Sprint)
1. P2-1: Nightly 3σ coverage regression
2. P2-6: Unit tests for gate/identity
3. P2-3: Plan artifact upload
4. Nightly L3: Option B (substitute tests)

### Medium-term (Next Month)
1. P2-7: Workflow dispatch improvements
2. P2-2: Full artifact identity validation
3. P2-5: Setup optimization
4. P2-8: Coverage reporting enhancements

### Long-term (Future)
1. Nightly L3: Option A (full FA/benchmark infrastructure)
2. P2-4: L4 release workflow
3. Cross-repo integration
4. Model validation

---

## Summary

**Nightly "全量" 问题**: 
- 当前行为正确：L3 的 2 个组都被 blocked
- 不是bug，是功能缺失
- 建议：短期接受 L2 为 "nightly full"，中期添加替代 L3 测试

**需要增加解决的问题** (按优先级):
1. **P2-1** (⚠️  IMPORTANT): Nightly 3σ coverage regression - 2-3h
2. **P2-6** (HIGH): Gate/identity unit tests - 3-4h
3. **P2-3** (MEDIUM): Plan artifact upload - 30min
4. **P2-2** (MEDIUM): Full artifact identity - 1-2h
5. **P2-7** (LOW): Workflow dispatch - 2h
6. **P2-5** (LOW): Setup optimization - 2-3h
7. **P2-8** (LOW): Coverage reporting - 2-3h
8. **P2-4** (FUTURE): L4 release - 40+ hours

**总预计时间** (P2-1 到 P2-8): 13-19 hours

**集成测试** (必须完成): 8 个场景，预计 2-4 hours

**Nightly L3 实现**:
- Option A: 16-24 hours
- Option B: 6-8 hours  
- Option C: 1 hour (文档更新)
