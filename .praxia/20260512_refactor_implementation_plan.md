# Implementation Plan: 7-Fixer ModelInputs Integration

## Executive Summary

Implement ModelInputs protocol migration and executor integration across 7 atomic fixers, sequenced to minimize dependencies and maximize parallelization. **Estimated 870 LOC total, 15-18 hours serial (10-12 hours with 2-fixer parallelization).**

## Context

**Affected Files:**
- `aminx/src/model_inputs.py` — Tier 1/2 protocol definitions (Fixer 1)
- `aminx/src/protocols.py` — TransformFn, FeaturizeFn aliases (Fixer 1)
- `aminx/src/pipeline_fns.py` — Registry extensions (Fixer 2)
- `aminx/src/pipeline_registry.py` — StageSet, PipelineFns shim (Fixer 2)
- `aminx/src/model/mpnn.py` — Model.stage_schema() impl (Fixer 3)
- `aminx/src/model/ligand_mpnn.py` — Model.stage_schema() + encoder_state_fn (Fixer 3, Fixer 5)
- `aminx/src/executor/base.py` — Executor.stage_set wiring (Fixer 4)
- `aminx/src/executor/*.py` — 3 executor implementations (Fixer 4)
- `aminx/src/mpnn_scoring_state_vmap_exact_ligand.py` — encoder_state_fn refactor (Fixer 5)
- `aminx/tests/model/test_stageset.py` — New StageSet unit tests (Fixer 6)
- `aminx/tests/executor/test_executor_integration.py` — Executor integration tests (Fixer 6)
- `aminx/tests/parity/test_*.py` — Parity test migration (Fixer 7)

## Phases & Dependencies

### Phase 1: Foundations (Protocols & Aliases) — Fixer 1.1
**Objective:** Establish Tier 1/2 protocol layer and type aliases, unblocking all downstream fixers.

**Task 1.1:** Add Tier 1 Protocol: TransformFn. Tier 2 Aliases: FeaturizeFn, EncoderFn. New concrete types: StageInput, StageOutput, StageSchema.
- Files: `model_inputs.py`, `protocols.py`
- Complexity: medium
- Effort: ~120 LOC, ~1.5 hours
- Depends on: none
- Verification: `pytest aminx/tests/ -k 'not parity_heavy' -q` (no import errors); `mypy aminx/src/{model_inputs,protocols}.py` (clean)

### Phase 2: Registry & StageSet (Unblock Model & Executor) — Fixer 2.1
**Objective:** Extend registry to support StageSet and PipelineFns shim.

**Task 2.1:** Add StageSet class with register/get_stage helpers. Add PipelineFns shim. 
- Files: `pipeline_registry.py`, `pipeline_fns.py`
- Complexity: medium
- Effort: ~180 LOC, ~2 hours
- Depends on: Fixer 1.1
- Verification: `pytest aminx/tests/ -k 'not parity_heavy' -q`; `mypy aminx/src/pipeline_*.py` (clean)

**⚠️ FIX #1 (Fixer 4.1 parallelization):** Explicitly state whether Fixer 4.1 includes validation against Model.stage_schema(). **This step ONLY wires the stage_set field onto the executor base class; validation lives in Fixer 4.2.** If validation happens in 4.1, add dependency edge Fixer 3 → Fixer 4.1.

### Phase 3: Model Schema Implementations (Parallel with Executor) — Fixers 3.1 & 3.2
**Objective:** Implement Model.stage_schema() for MPNN and LigandMPNN variants.

**Task 3.1:** Implement Model.stage_schema() in MPNN class.
- Files: `mpnn.py`
- Complexity: medium
- Effort: ~60 LOC, ~1 hour
- Depends on: Fixer 2.1
- Verification: `pytest aminx/tests/model/ -k 'not parity_heavy' -q`

**Task 3.2:** Implement Model.stage_schema() in LigandMPNN class (encoder_state_fn placeholder).
- Files: `ligand_mpnn.py`
- Complexity: medium
- Effort: ~60 LOC, ~1 hour
- Depends on: Fixer 2.1
- Verification: `pytest aminx/tests/model/ -k 'test_ligand' -q`

### Phase 4: Executor Wiring (Parallel with Model, Follows Registry) — Fixers 4.1 & 4.2
**Objective:** Wire Executor.stage_set parameter and validate stage names.

**Task 4.1:** Add stage_set parameter and validation to Executor base class.
- Files: `executor/base.py`
- Complexity: high
- Effort: ~60 LOC, ~1.5 hours
- Depends on: Fixer 2.1
- Verification: `pytest aminx/tests/executor/ -k 'not integration' -q`
- **Note:** Does NOT call or validate against Model.stage_schema(). All validation logic lives in Fixer 4.2.

**Task 4.2:** Implement stage_set wiring for 3 executor variants.
- Files: `executor/{sequential,parallel,gpu}.py`
- Complexity: medium
- Effort: ~100 LOC, ~1.5 hours
- Depends on: Fixer 4.1
- Verification: `pytest aminx/tests/executor/ -k 'not integration' -q`

**⚠️ FIX #2 (PipelineFns backward-compatibility):** Add explicit guarantee: **All existing public methods (default(), from_callables(), resolve_logit_transform(), resolve_ar_logit_transform(), resolve_encoder_state_fn()) are preserved with identical signatures and return types. No breaking changes to the public API.** If any method changes, add Fixer 2b to update all consumer call sites.

### Phase 5: LigandMPNN Encoder State Threading — Fixers 5.1 & 5.2
**Objective:** Refactor encoder_state_fn to TransformFn contract and thread through LigandMPNN.

**Task 5.1:** Refactor encoder_state_fn to match TransformFn signature. Thread into LigandMPNN.stage_schema().
- Files: `ligand_mpnn.py`
- Complexity: high
- Effort: ~70 LOC, ~1.5 hours
- Depends on: Fixer 3.2, Fixer 4.2
- Verification: `pytest aminx/tests/model/ -k 'ligand' -q`

**Task 5.2:** Update encoder_state_fn call site in scoring vmap.
- Files: `mpnn_scoring_state_vmap_exact_ligand.py`
- Complexity: medium
- Effort: ~50 LOC, ~1 hour
- Depends on: Fixer 5.1
- Verification: `pytest aminx/tests/sampling/test_state_vmap_exact_jit.py -q` (no regression)

### Phase 6: Test Suite Expansion — Fixers 6.1 & 6.2
**Objective:** Add StageSet unit tests and Executor integration tests.

**Task 6.1:** Create StageSet unit test suite.
- Files: `tests/model/test_stageset.py` (new)
- Complexity: medium
- Effort: ~80 LOC, ~1.5 hours
- Depends on: Fixer 2.1
- Verification: `pytest aminx/tests/model/test_stageset.py -v` (coverage >90%)

**Task 6.2:** Create Executor integration test suite.
- Files: `tests/executor/test_executor_integration.py` (new)
- Complexity: medium
- Effort: ~100 LOC, ~2 hours
- Depends on: Fixer 4.2, Fixer 3.2
- Verification: `pytest aminx/tests/executor/test_executor_integration.py -v` (coverage >85%)

**⚠️ FIX #3 (Tier 1/Tier 2 mapping):** Add explicit mapping table:
- `LogitTransformFn = FuseFn[Float[Array, "S L V"], Float[Array, "L V"]]` (reduce across S)
- `ARLogitTransformFn = FuseFn[Float[Array, "S V"], Float[Array, "V"]]` (reduce across S)
- `EncoderStateFn = RollingFn[Any, BackboneGeometry, EncoderOutput]` (carry-based scan)

### Phase 7: Parity Test Migration — Fixer 7.1
**Objective:** Migrate existing parity tests to use new Executor path.

**Task 7.1:** Update parity test fixtures to instantiate Executor with model.stage_schema().
- Files: `tests/parity/test_ligandmpnn_equivalence.py`, `tests/parity/test_mpnn_equivalence.py`
- Complexity: medium
- Effort: ~80 LOC, ~1.5 hours
- Depends on: Fixer 5.2, Fixer 6.2
- Verification: `pytest aminx/tests/parity/ -m parity_heavy -v` (all pass within tolerance atol=1e-5)

**⚠️ FIX #4 (LOC delta reconciliation):** Spec LOC estimate (250-400 LOC, excludes docstrings/type stubs/comments) is narrower than plan LOC (~870 LOC, which includes full docstrings, type annotations, and error messages). Discrepancy source:
- Spec LOC is conservative (doesn't include docstrings, type stubs, comments)
- Plan LOC includes full docstrings, type annotations, error messages
- Clarification: Plan LOC is accurate; use as basis for effort estimation

**⚠️ FIX #5 (Registry fixture enumeration):** registry_snapshot fixture in tests/pipeline/conftest.py must handle:
- `pipeline_registry._REGISTRY` (main hook registry)
- `pipeline_fns.DEFAULT_FEATURIZE_UID`, `DEFAULT_ENCODE_UID`, `DEFAULT_DECODE_UID` (sentinel constants)
- Any cloudpickle-hashed UID keys registered during test

**⚠️ FIX #6 (Parity baseline capture):** Add pre-flight step to pre_flight_checklist:
- `[ ] PYTHONPATH=aminx/src uv run pytest aminx/tests/parity/ -q 2>&1 | tee /tmp/parity_baseline.log` (baseline parity pass before Fixer 1 starts)
- Before Fixer 7 merge: `pytest aminx/tests/parity/ -q 2>&1 | diff /tmp/parity_baseline.log -` (must be identical within atol=1e-5)

## Parallel Execution Groups

```
Group 1: [1.1]                  (sequential dependency)
Group 2: [2.1]                  (follows 1.1)
Group 3: [3.1, 3.2, 4.1]        (all depend on 2.1; can run in parallel)
Group 4: [4.2]                  (follows 4.1)
Group 5: [5.1, 6.1]             (5.1 depends on 3.2+4.2; 6.1 depends on 2.1; can run in parallel)
Group 6: [5.2, 6.2]             (5.2 depends on 5.1; 6.2 depends on 4.2+3.2; can run in parallel)
Group 7: [7.1]                  (depends on 5.2+6.2)
```

**Critical path:** 1.1 → 2.1 → 4.1 → 4.2 → 5.1 → 5.2 → 7.1 (7 sequential steps, ~10-12 hours)

**With parallelization:** 1.1 → 2.1 → {3,4.1} → 4.2 → {5.1,6.1} → {5.2,6.2} → 7.1 (reduces wall-clock to ~10 hours)

## Effort Estimate (Per Fixer)

| Fixer | LOC  | Hours | Task |
|-------|------|-------|------|
| 1.1   | 120  | 1.5   | Add Tier 1/2 protocols + type aliases |
| 2.1   | 180  | 2.0   | Extend registry with StageSet + PipelineFns shim |
| 3.1   | 60   | 1.0   | Model.stage_schema() for MPNN |
| 3.2   | 60   | 1.0   | Model.stage_schema() for LigandMPNN |
| 4.1   | 60   | 1.5   | Add stage_set parameter to Executor base |
| 4.2   | 100  | 1.5   | Implement stage_set wiring for 3 executor variants |
| 5.1   | 70   | 1.5   | Refactor encoder_state_fn to TransformFn signature |
| 5.2   | 50   | 1.0   | Update encoder_state_fn call site in scoring vmap |
| 6.1   | 80   | 1.5   | StageSet unit test suite |
| 6.2   | 100  | 2.0   | Executor integration test suite |
| 7.1   | 80   | 1.5   | Parity test migration |
| **TOTAL** | **870** | **15-18** | **Serial (10-12 with parallelization)** |

## Pre-Flight Checklist

- [ ] `git status` clean; no staged or unstaged changes
- [ ] `git log` shows latest main commit
- [ ] `PYTHONPATH=aminx/src; uv run pytest aminx/tests/sampling/ -k 'not parity_heavy' -q` passes (baseline green)
- [ ] `mypy aminx/src/ --no-error-summary` returns 0 errors (baseline clean)
- [ ] No local branch; main is checked out and up-to-date with origin/main
- [ ] `.praxia/` directory exists and is writable (for OODA log)
- [ ] `aminx/tests/model/`, `tests/executor/`, `tests/parity/` directories exist
- [ ] **NEW:** `PYTHONPATH=aminx/src uv run pytest aminx/tests/parity/ -q 2>&1 | tee /tmp/parity_baseline.log` (baseline parity pass before Fixer 1)

## Go/No-Go Criteria

- All 7 fixers committed with passing CI (mypy, pytest, coverage thresholds)
- Zero mypy errors in `aminx/src/`
- All automated test suites green (smoke, model, executor, parity)
- Executor + Model.stage_schema() contract verified: identical inputs → identical outputs vs direct model call
- Code review approval on Fixer 5 (encoder_state_fn refactor) and Fixer 7 (parity migration)
- No regressions in existing downstream code
- Parity baseline unchanged within tolerance (atol=1e-5)

## Rollback Strategy

| Failure | Action |
|---------|--------|
| Fixer 1 | `git revert <fixer-1 commit>`; re-run after fix |
| Fixer 2 | `git revert <fixer-2 commit>`; fixers 3-7 blocked; re-run 3-7 |
| Fixer 3 | `git revert <fixer-3a/3b>`; fixers 4-7 blocked; re-run 4-7 |
| Fixer 4 | `git revert <fixer-4a/4b>`; fixers 5-7 blocked; re-run 5-7 |
| Fixer 5 | `git revert <fixer-5a/5b>`; fixer 7 blocked; re-run 7 |
| Fixer 6 | `git revert <fixer-6a/6b>`; no downstream dependency |
| Fixer 7 | `git revert <fixer-7>`; no downstream dependency |
