# Architecture Migration Summary - For Fresh Agent

**Date**: November 3, 2025  
**Task**: Unify `eqx_new.py` as primary implementation  
**Priority**: HIGH  
**Estimated Time**: 2-3 days

---

## TL;DR

We have 3 implementations (functional, old eqx, new eqx). New eqx (`eqx_new.py`) is complete and equivalent. Need to migrate all code to use it while keeping tests passing.

---

## Current Status ✅

**What's Working:**
- ✅ New Equinox model (`eqx_new.PrxteinMPNN`) - 4/4 equivalence tests passing
- ✅ HuggingFace deployment - 8 models in `.eqx` format, 15/15 loading tests passing
- ✅ New `load_model()` API in `io/weights.py`
- ✅ All functional tests passing

**What Needs Migration:**
- ⏳ User-facing APIs (`run/sampling.py`, `run/scoring.py`)
- ⏳ Internal utilities (`sampling/`, `scoring/`)
- ⏳ Tests that use functional implementation
- ⏳ Documentation and examples

---

## Architecture Map

```
Current (Functional):
  User API → functional/model.py → Pure JAX functions

Target (Equinox):
  User API → eqx_new.PrxteinMPNN → JAX + Equinox
```

---

## Migration Strategy (Phased Approach)

### Phase 1: Adapter Layer ⭐ START HERE
**Goal**: Bridge between new and old without breaking anything

**Action**: Update `src/prxteinmpnn/functional/model.py`:

```python
def get_functional_model(
    model_version="v_48_020",
    model_weights="original",
    use_new_architecture=True,  # Feature flag for gradual rollout
):
    if use_new_architecture:
        from prxteinmpnn.io.weights import load_model
        return load_model(model_version, model_weights)
    else:
        return _load_legacy_functional_params(...)
```

**Why**: Allows tests to run with both architectures in parallel.

### Phase 2: Update Run Utilities
**Files**: `run/prep.py`, `run/sampling.py`, `run/scoring.py`

**Changes**:
- Return type: `ModelParameters` → `PrxteinMPNN`
- Function calls: `functional_encoder(params, ...)` → `model._call_unconditional(...)`
- Add adapter functions that work with both types during migration

### Phase 3: Test Everything
**Strategy**: Parametrize tests to run with both architectures

```python
@pytest.mark.parametrize("use_new_arch", [False, True])
def test_sampling(use_new_arch):
    model = get_functional_model(use_new_architecture=use_new_arch)
    # Test remains the same
```

### Phase 4: Flip the Switch
- Set `use_new_architecture=True` as default
- Add deprecation warnings for legacy usage
- Monitor for issues

### Phase 5: Clean Up
- Remove old `eqx.py` (not eqx_new!)
- Archive `functional/` for reference
- Update all docs

---

## Step-by-Step Implementation

### 1. Verify Everything Works (5 min)
```bash
uv run pytest tests/test_eqx_equivalence.py -v  # Should pass 4/4
uv run pytest tests/io/test_hf_loading.py -v    # Should pass 15/15
```

### 2. Create Adapter (30 min)
- File: `src/prxteinmpnn/functional/model.py`
- Add `use_new_architecture` parameter
- Test both paths work

### 3. Update Type Hints (1 hour)
- File: `src/prxteinmpnn/utils/types.py`
- Add: `Model = PrxteinMPNN | ModelParameters`
- Update all type hints in `run/` directory

### 4. Create Wrapper Functions (2 hours)
- File: `src/prxteinmpnn/sampling/adapter.py` (new)
- Functions that work with both PyTree and PrxteinMPNN
- Example: `call_model_unconditional(model, ...)`

### 5. Update Run Utils (2 hours)
- Update `prep.py` to return `PrxteinMPNN`
- Update `sampling.py` and `scoring.py` to use new model
- Use adapter functions for compatibility

### 6. Migrate Tests (3 hours)
- Add parametrization to run tests with both architectures
- Debug failures one by one
- Ensure 100% test retention

### 7. Benchmark (1 hour)
- Compare performance old vs new
- Must be within 5% of baseline

### 8. Documentation (1 hour)
- Update README with new examples
- Create migration guide
- Update all docstrings

---

## Key Files Reference

**Core Implementation:**
- `src/prxteinmpnn/eqx_new.py` - New unified Equinox model ✅
- `src/prxteinmpnn/io/weights.py` - Model loading ✅
- `src/prxteinmpnn/functional/model.py` - TO UPDATE (adapter)

**Run Utilities:**
- `src/prxteinmpnn/run/prep.py` - TO UPDATE
- `src/prxteinmpnn/run/sampling.py` - TO UPDATE
- `src/prxteinmpnn/run/scoring.py` - TO UPDATE

**Tests:**
- `tests/test_eqx_equivalence.py` - Equivalence proofs ✅
- `tests/io/test_hf_loading.py` - HuggingFace loading ✅
- `tests/run/` - TO MIGRATE

**Documentation:**
- `docs/ARCHITECTURE_MIGRATION_PLAN.md` - Full detailed plan
- `docs/HF_DEPLOYMENT.md` - Recent HuggingFace work
- `docs/EQUINOX_FUNCTIONAL_EQUIVALENCE.md` - Equivalence details

---

## Critical Requirements

### Must Maintain:
- ✅ All tests passing (100% retention)
- ✅ Performance within 5% of baseline
- ✅ Backward compatibility (users don't break)
- ✅ Code quality (Ruff, Pyright strict)
- ✅ DRY principles

### Standards:
- Google-style docstrings
- Type hints everywhere
- JAX-compatible (jit, vmap, scan)
- No Python loops in hot paths

---

## Risk Mitigation

**Feature Flag Approach:**
- Allows gradual rollout
- Easy rollback if issues
- Tests run in parallel
- Low risk to users

**Testing Strategy:**
- Run all tests with both architectures
- Compare outputs for equivalence
- Benchmark performance
- Only remove legacy code after stable

---

## Success Criteria

- [ ] All existing tests pass with `use_new_architecture=True`
- [ ] Performance equal or better
- [ ] No Pyright errors
- [ ] No Ruff warnings
- [ ] Documentation updated
- [ ] Migration guide created

---

## Quick Commands

```bash
# Run equivalence tests
uv run pytest tests/test_eqx_equivalence.py -v

# Run HuggingFace tests
uv run pytest tests/io/test_hf_loading.py -v

# Run all functional tests
uv run pytest tests/functional/ -v

# After changes - run everything
uv run pytest tests/ -v

# Lint check
uv run ruff check src/ --fix

# Type check
uv run pyright src/
```

---

## Questions? Check These First

1. **"What's the difference between eqx.py and eqx_new.py?"**
   - `eqx.py` = old legacy wrapper (incomplete)
   - `eqx_new.py` = new complete implementation ✅ Use this!

2. **"Why not just delete functional code?"**
   - Too risky - it's used everywhere
   - Gradual migration is safer
   - Keeps tests running during transition

3. **"What if tests fail with new architecture?"**
   - Debug incrementally using adapter functions
   - Compare outputs with functional version
   - Use `use_new_architecture=False` to isolate issues

4. **"How do I know if migration is successful?"**
   - All tests pass with `use_new_architecture=True`
   - Performance benchmarks acceptable
   - No deprecation warnings in tests

---

## Contact & Support

**Read First:**
- Full plan: `docs/ARCHITECTURE_MIGRATION_PLAN.md`
- Dev standards: `AGENTS.md`
- Recent work: `docs/HF_DEPLOYMENT.md`

**Got Stuck?**
1. Check if equivalence tests still pass
2. Verify HuggingFace models load correctly  
3. Review adapter function implementation
4. Ask questions early - architecture is critical!

---

**Ready to Start?** → Begin with Step 2 (Create Adapter Function)

Good luck! 🚀
