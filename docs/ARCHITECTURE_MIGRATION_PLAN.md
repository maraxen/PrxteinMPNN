# Architecture Migration Plan: Unifying eqx_new.py as the Primary Implementation

**Date**: November 3, 2025  
**Status**: Planning Phase  
**Priority**: HIGH  
**Complexity**: MEDIUM-HIGH

---

## Executive Summary

This document provides a comprehensive plan for migrating PrxteinMPNN from a dual-architecture system (functional + old Equinox) to a unified architecture using `eqx_new.py` as the single source of truth. The migration must be performed carefully to maintain all existing functionality, tests, and user-facing APIs while improving code quality and maintainability.

---

## Current State Analysis

### Architecture Overview

**Three Parallel Implementations Currently Exist:**

1. **`functional/`** - Original pure-functional JAX implementation
   - Used by: `run/` utilities, most tests, conversion scripts
   - Status: Stable, well-tested, performance-optimized
   - File: `src/prxteinmpnn/functional/model.py`
   - Function: `get_functional_model()` - returns PyTree of parameters

2. **`eqx.py`** - Old Equinox wrapper (legacy)
   - Status: To be deprecated
   - Uses: `PrxteinMPNNWrapped` class
   - Issues: Incomplete, not numerically equivalent

3. **`eqx_new.py`** - New unified Equinox implementation
   - Status: ✅ Complete, numerically equivalent (4/4 tests passing)
   - Class: `PrxteinMPNN`
   - Features: Full encoder/decoder, all decoding modes
   - Tests: `tests/test_eqx_equivalence.py` (11.6s, rtol=1e-5)

### Current Dependencies

```
User-Facing APIs (run/)
    ↓
functional/model.py (get_functional_model)
    ↓
Functional PyTree Parameters
    ↓
Pure JAX functions (encoder, decoder, etc.)
```

### HuggingFace Model Status

✅ **Recently Completed:**
- 8 models uploaded in `.eqx` format to `maraxen/prxteinmpnn`
- New `io/weights.py` with `load_model()` high-level API
- Comprehensive test suite in `tests/io/test_hf_loading.py`
- Models stored at: `https://huggingface.co/maraxen/prxteinmpnn/tree/main/eqx`

### Test Coverage Status

**Passing Tests:**
- ✅ `tests/test_eqx_equivalence.py` - 4/4 core equivalence tests
- ✅ `tests/io/test_hf_loading.py` - 15/15 HuggingFace loading tests
- ✅ `tests/functional/` - All functional tests passing
- ✅ `tests/ensemble/` - Most passing (2 backlogged)

**Tests Using Functional:**
- `tests/run/` - All sampling/scoring tests
- `tests/scoring/` - Score calculation tests
- `tests/sampling/` - Sampling strategy tests

---

## Migration Goals

### Primary Objectives

1. **Unify Architecture**: Make `eqx_new.PrxteinMPNN` the single implementation
2. **Maintain Compatibility**: Keep all existing APIs working
3. **Preserve Tests**: Ensure all tests pass with new architecture
4. **Improve Performance**: Leverage Equinox optimizations
5. **Simplify Codebase**: Remove redundant implementations

### Success Criteria

- [ ] All existing tests pass (100% retention)
- [ ] User-facing APIs unchanged (backward compatible)
- [ ] Performance equal or better than functional
- [ ] Code coverage maintained or improved
- [ ] Documentation updated
- [ ] Migration path for downstream users

---

## Migration Strategy

### Phase 1: Adapter Layer (Recommended Approach)

**Goal**: Create a bridge between new Equinox models and functional API

**Why This Approach:**
- ✅ Minimal disruption to existing code
- ✅ Allows gradual migration
- ✅ Easy rollback if issues arise
- ✅ Tests can run in parallel (old + new)

**Implementation:**

```python
# src/prxteinmpnn/functional/model.py

from typing import Literal
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.eqx_new import PrxteinMPNN

def get_functional_model(
    model_version: Literal["v_48_002", "v_48_010", "v_48_020", "v_48_030"] = "v_48_020",
    model_weights: Literal["original", "soluble"] = "original",
    use_new_architecture: bool = True,  # Feature flag
) -> PrxteinMPNN | dict:
    """Load a ProteinMPNN model.
    
    Args:
        model_version: Model version to load
        model_weights: Weight type (original or soluble)
        use_new_architecture: If True, returns new PrxteinMPNN model.
                             If False, returns legacy functional PyTree.
    
    Returns:
        Either a PrxteinMPNN model (new) or parameter PyTree (legacy)
    """
    if use_new_architecture:
        # New path: Load from HuggingFace as Equinox model
        return load_model(model_version=model_version, model_weights=model_weights)
    else:
        # Legacy path: Load as functional PyTree
        return _load_legacy_functional_model(model_version, model_weights)
```

### Phase 2: Update Run Utilities

**Files to Update:**
- `src/prxteinmpnn/run/prep.py`
- `src/prxteinmpnn/run/sampling.py`
- `src/prxteinmpnn/run/scoring.py`

**Changes Needed:**

1. **Update `prep.py`:**

```python
# Current:
def prep_protein_stream_and_model(spec: Specs) -> tuple[IterDataset, ModelParameters]:
    model_parameters = get_functional_model(
        model_version=spec.model_version,
        model_weights=spec.model_weights,
    )
    return protein_iterator, model_parameters

# New:
def prep_protein_stream_and_model(spec: Specs) -> tuple[IterDataset, PrxteinMPNN]:
    model = get_functional_model(  # Returns PrxteinMPNN now
        model_version=spec.model_version,
        model_weights=spec.model_weights,
        use_new_architecture=True,
    )
    return protein_iterator, model
```

2. **Update sampling/scoring functions:**
   - Replace functional calls with `model()` calls
   - Use `model._call_unconditional()`, `model._call_conditional()`, etc.
   - Update type hints from `ModelParameters` to `PrxteinMPNN`

### Phase 3: Update Sampling/Scoring Modules

**Files to Update:**
- `src/prxteinmpnn/sampling/sample.py`
- `src/prxteinmpnn/scoring/score.py`

**Strategy:**
- Create wrapper functions that accept both PyTree and PrxteinMPNN
- Use `isinstance()` checks to dispatch to correct implementation
- Gradually migrate to Equinox-only versions

**Example:**

```python
def make_sample_sequences(
    model: PrxteinMPNN | dict,  # Accept both types
    decoding_order_fn,
    sampling_strategy,
):
    if isinstance(model, PrxteinMPNN):
        # Use new Equinox model
        def sample_fn(key, X, E, mask, ...):
            return model(
                structure_coordinates=X,
                mask=mask,
                residue_index=residue_idx,
                chain_index=chain_idx,
                decoding_approach="autoregressive",
                prng_key=key,
                temperature=temperature,
            )
        return sample_fn
    else:
        # Legacy functional path
        return _make_sample_sequences_functional(model, ...)
```

### Phase 4: Test Migration

**Approach:**
1. Run tests with `use_new_architecture=False` (baseline)
2. Run tests with `use_new_architecture=True` (migration target)
3. Compare outputs, debug discrepancies
4. Add new tests for Equinox-specific features

**Test Files to Update:**
- `tests/run/test_sampling.py`
- `tests/run/test_scoring.py`
- `tests/scoring/test_score.py`
- `tests/sampling/test_sample.py`

**Testing Strategy:**
```python
# Add parametrize fixture to test both architectures
@pytest.mark.parametrize("use_new_arch", [False, True])
def test_sampling(use_new_arch):
    model = get_functional_model(use_new_architecture=use_new_arch)
    # ... rest of test
```

### Phase 5: Deprecation & Cleanup

**After all tests pass with new architecture:**

1. Set `use_new_architecture=True` as default
2. Add deprecation warnings for functional PyTree usage
3. Remove old `eqx.py` (legacy wrapper)
4. Archive `functional/` implementations (keep for reference)
5. Update all documentation

---

## Implementation Plan

### Step-by-Step Instructions for Fresh Agent

#### Step 1: Environment Setup (5 min)

```bash
cd /Users/mar/MIT\ Dropbox/Marielle\ Russo/2025_workspace/PrxteinMPNN
uv sync
uv run pytest tests/test_eqx_equivalence.py -v  # Should pass 4/4
uv run pytest tests/io/test_hf_loading.py -v  # Should pass 15/15
```

**Verify:**
- All core equivalence tests passing
- HuggingFace models loadable
- No import errors

#### Step 2: Create Adapter Function (30 min)

**File**: `src/prxteinmpnn/functional/model.py`

**Tasks:**
1. Add `use_new_architecture` parameter to `get_functional_model()`
2. Import `load_model` from `prxteinmpnn.io.weights`
3. Add conditional logic for new vs legacy loading
4. Add deprecation warning when `use_new_architecture=False`
5. Update docstring with migration guidance

**Test:**
```python
# Test both paths work
from prxteinmpnn.functional import get_functional_model

# Legacy
params = get_functional_model(use_new_architecture=False)
assert isinstance(params, dict)

# New
model = get_functional_model(use_new_architecture=True)
from prxteinmpnn.eqx_new import PrxteinMPNN
assert isinstance(model, PrxteinMPNN)
```

#### Step 3: Update Type Hints (1 hour)

**Files to update:**
- `src/prxteinmpnn/utils/types.py` - Add `Model` type alias
- `src/prxteinmpnn/run/prep.py` - Update return types
- `src/prxteinmpnn/run/sampling.py` - Update parameter types
- `src/prxteinmpnn/run/scoring.py` - Update parameter types

**Example changes:**

```python
# src/prxteinmpnn/utils/types.py
from typing import TypeAlias
from prxteinmpnn.eqx_new import PrxteinMPNN

# Support both during migration
ModelParameters: TypeAlias = dict[str, Any]  # Legacy
Model: TypeAlias = PrxteinMPNN | ModelParameters  # Union type
```

#### Step 4: Create Wrapper Functions (2-3 hours)

**File**: `src/prxteinmpnn/sampling/adapter.py` (new file)

**Purpose**: Bridge between functional and Equinox interfaces

```python
"""Adapter functions for gradual migration to Equinox architecture."""

from typing import Protocol
from prxteinmpnn.eqx_new import PrxteinMPNN

def call_model_unconditional(
    model: PrxteinMPNN | dict,
    structure_coords,
    mask,
    residue_index,
    chain_index,
    **kwargs
):
    """Call model in unconditional mode (works with both architectures)."""
    if isinstance(model, PrxteinMPNN):
        edge_features, neighbor_indices, _ = model.features(
            jax.random.PRNGKey(0),
            structure_coords,
            mask,
            residue_index,
            chain_index,
            backbone_noise=kwargs.get('backbone_noise', 0.0),
        )
        return model._call_unconditional(edge_features, neighbor_indices, mask)
    else:
        # Call functional version
        return call_functional_unconditional(model, structure_coords, mask, ...)
```

**Similar wrappers needed for:**
- `call_model_conditional()`
- `call_model_autoregressive()`
- `extract_features()`

#### Step 5: Update Run Utilities (2 hours)

**File**: `src/prxteinmpnn/run/prep.py`

```python
# Change return type
def prep_protein_stream_and_model(
    spec: Specs
) -> tuple[IterDataset, PrxteinMPNN]:  # Changed from ModelParameters
    """Prepare the protein data stream and model.
    
    Now returns Equinox PrxteinMPNN model instead of functional parameters.
    """
    ...
    model = get_functional_model(
        model_version=spec.model_version,
        model_weights=spec.model_weights,
        use_new_architecture=True,  # Default to new
    )
    return protein_iterator, model
```

**File**: `src/prxteinmpnn/run/sampling.py`

- Update all function signatures to accept `PrxteinMPNN`
- Replace functional calls with model method calls
- Update vmapped functions to work with Equinox

**File**: `src/prxteinmpnn/run/scoring.py`

- Same updates as sampling.py
- Ensure scoring uses `_call_unconditional` or `_call_conditional`

#### Step 6: Add Backward Compatibility Tests (1 hour)

**File**: `tests/test_architecture_compatibility.py` (new)

```python
"""Test that both architectures produce same results during migration."""

import pytest
import jax.numpy as jnp
from prxteinmpnn.functional import get_functional_model

@pytest.mark.parametrize("use_new_arch", [False, True])
def test_model_loading(use_new_arch):
    """Test model loads correctly with both architectures."""
    model = get_functional_model(
        model_version="v_48_020",
        use_new_architecture=use_new_arch,
    )
    assert model is not None

def test_output_equivalence():
    """Test that old and new architectures produce same outputs."""
    # Load both
    legacy_params = get_functional_model(use_new_architecture=False)
    new_model = get_functional_model(use_new_architecture=True)
    
    # Create test input
    # ... (reuse from test_eqx_equivalence.py)
    
    # Compare outputs
    # legacy_out = call_functional(legacy_params, ...)
    # new_out = new_model(...)
    # assert jnp.allclose(legacy_out, new_out, rtol=1e-5)
```

#### Step 7: Update Existing Tests (3-4 hours)

**Strategy**: Add feature flag to all tests

**Files**:
- `tests/run/test_sampling.py`
- `tests/run/test_scoring.py`
- `tests/scoring/test_score.py`
- `tests/sampling/test_sample.py`

**Approach**:
1. Add fixture for architecture selection
2. Run each test with both architectures
3. Debug failures one by one
4. Document any known differences

**Example**:

```python
@pytest.fixture(params=[False, True], ids=["functional", "equinox"])
def model(request):
    """Fixture that provides both architectures."""
    return get_functional_model(
        model_version="v_48_020",
        use_new_architecture=request.param,
    )

def test_sampling(model):
    """Test sampling works with both architectures."""
    # Test code remains the same
    # Adapter functions handle the differences
```

#### Step 8: Performance Benchmarking (1 hour)

**Create**: `benchmarks/architecture_comparison.py`

```python
"""Benchmark old vs new architecture."""

import time
import jax
from prxteinmpnn.functional import get_functional_model

def benchmark_inference(use_new_arch, num_iterations=100):
    model = get_functional_model(use_new_architecture=use_new_arch)
    # ... setup inputs
    
    start = time.time()
    for _ in range(num_iterations):
        # ... run inference
        pass
    end = time.time()
    
    return end - start

# Compare
legacy_time = benchmark_inference(False)
new_time = benchmark_inference(True)
print(f"Legacy: {legacy_time:.3f}s, New: {new_time:.3f}s")
print(f"Speedup: {legacy_time/new_time:.2f}x")
```

#### Step 9: Update Documentation (1-2 hours)

**Files to update:**
1. `README.md` - Update examples to use new API
2. `docs/QUICK_START.md` - Migration guide
3. `docs/API.md` - Document new interfaces
4. Docstrings in all updated modules

**Create new docs:**
1. `docs/MIGRATION_GUIDE.md` - For downstream users
2. `docs/ARCHITECTURE.md` - New unified architecture

#### Step 10: Gradual Rollout (ongoing)

**Week 1:**
- [ ] Adapter functions working
- [ ] Tests run with both architectures
- [ ] Performance benchmarks complete

**Week 2:**
- [ ] Set `use_new_architecture=True` as default
- [ ] Monitor for issues
- [ ] Add deprecation warnings

**Week 3:**
- [ ] Remove legacy code
- [ ] Clean up temporary adapter functions
- [ ] Archive functional implementation

---

## Risk Mitigation

### Known Risks

1. **Performance Regression**
   - Mitigation: Comprehensive benchmarks before/after
   - Rollback: Keep feature flag for easy revert

2. **Test Failures**
   - Mitigation: Gradual migration with parallel tests
   - Rollback: Fix issues before removing legacy code

3. **API Breaking Changes**
   - Mitigation: Maintain backward compatibility layer
   - Rollback: Version bump only after stable

4. **Downstream User Impact**
   - Mitigation: Clear migration guide and deprecation timeline
   - Support: Provide migration assistance

### Testing Checklist

- [ ] All existing tests pass with new architecture
- [ ] No performance regression (within 5%)
- [ ] Memory usage unchanged or improved
- [ ] JIT compilation times acceptable
- [ ] Batch processing works correctly
- [ ] All sampling strategies work
- [ ] All scoring modes work
- [ ] HDF5 streaming still functional

---

## Code Quality Requirements

### Standards to Maintain

1. **Type Safety**
   - All functions properly type-hinted
   - Pyright strict mode passing
   - No `Any` types except where necessary

2. **Testing**
   - 100% test retention
   - New tests for Equinox-specific features
   - Integration tests for full pipeline

3. **Documentation**
   - Google-style docstrings
   - Usage examples in docstrings
   - Migration guide for users

4. **Performance**
   - JAX transformations (jit, vmap, scan) compatible
   - No unnecessary Python loops
   - Efficient memory usage

5. **DRY Principle**
   - No duplicate code between architectures
   - Shared utilities properly factored
   - Clear separation of concerns

### Code Review Checklist

Before merging:
- [ ] All tests passing
- [ ] Ruff linting clean
- [ ] Pyright type checking passing
- [ ] Documentation updated
- [ ] Performance benchmarks acceptable
- [ ] No breaking changes to public API
- [ ] Deprecation warnings in place

---

## Success Metrics

### Quantitative Goals

- ✅ 100% test retention (all existing tests pass)
- ✅ Performance within 5% of baseline
- ✅ <10% increase in memory usage
- ✅ <20% increase in compile time
- ✅ 0 new Pyright errors
- ✅ 0 new Ruff warnings

### Qualitative Goals

- ✅ Code is more maintainable
- ✅ Architecture is easier to understand
- ✅ Documentation is comprehensive
- ✅ Migration path is clear
- ✅ User feedback is positive

---

## Timeline Estimate

**Total: 2-3 days of focused work**

| Phase | Time | Dependencies |
|-------|------|--------------|
| Setup & Adapter | 2 hours | None |
| Wrappers | 3 hours | Adapter |
| Run Utils Update | 2 hours | Wrappers |
| Test Migration | 4 hours | Run Utils |
| Benchmarking | 1 hour | Tests passing |
| Documentation | 2 hours | Implementation complete |
| **Buffer** | 4 hours | Issues, debugging |
| **Total** | ~18 hours | |

---

## Next Steps for Fresh Agent

### Immediate Actions (Start Here)

1. **Read These Documents (30 min)**
   - This file (ARCHITECTURE_MIGRATION_PLAN.md)
   - `docs/HF_DEPLOYMENT.md` - Recent HuggingFace work
   - `docs/EQUINOX_FUNCTIONAL_EQUIVALENCE.md` - Equivalence testing
   - `AGENTS.md` - Development standards

2. **Verify Current State (15 min)**
   ```bash
   # All these should pass
   uv run pytest tests/test_eqx_equivalence.py -v
   uv run pytest tests/io/test_hf_loading.py -v
   uv run pytest tests/functional/ -v
   ```

3. **Create Feature Branch (5 min)**
   ```bash
   git checkout -b feature/unify-eqx-architecture
   ```

4. **Start with Step 2 (Adapter Function)**
   - Low risk, high value
   - Easy to test in isolation
   - Enables parallel development

### Questions to Ask

Before starting implementation:

1. Should we keep functional implementation for performance comparison?
2. What's the deprecation timeline for legacy API?
3. Are there any downstream users we need to coordinate with?
4. Should we version-bump to 0.2.0 after migration?

### Communication

**Status Updates**: Post progress in PR comments
**Blockers**: Flag immediately if tests start failing
**Questions**: Ask early and often - architecture decisions are critical

---

## Appendix

### File Structure After Migration

```
src/prxteinmpnn/
├── __init__.py              # Public API exports
├── eqx_new.py              # PRIMARY IMPLEMENTATION (rename to eqx.py?)
├── io/
│   └── weights.py          # ✅ Already updated with load_model()
├── run/
│   ├── prep.py             # TO UPDATE: model loading
│   ├── sampling.py         # TO UPDATE: use PrxteinMPNN
│   └── scoring.py          # TO UPDATE: use PrxteinMPNN
├── sampling/
│   ├── sample.py           # TO UPDATE: adapter functions
│   └── adapter.py          # NEW: bridge functional→Equinox
├── scoring/
│   └── score.py            # TO UPDATE: adapter functions
├── functional/             # ARCHIVE: keep for reference
│   └── model.py            # UPDATE: add adapter
└── utils/
    └── types.py            # UPDATE: new type aliases
```

### Key Type Definitions

```python
# Current (functional)
ModelParameters = dict[str, jnp.ndarray]

# New (Equinox)
from prxteinmpnn.eqx_new import PrxteinMPNN

# Union (during migration)
Model = PrxteinMPNN | ModelParameters
```

### References

- **Core Equivalence Tests**: `tests/test_eqx_equivalence.py`
- **HuggingFace Integration**: `tests/io/test_hf_loading.py`
- **Model Loading**: `src/prxteinmpnn/io/weights.py`
- **Equinox Implementation**: `src/prxteinmpnn/eqx_new.py`
- **Functional Implementation**: `src/prxteinmpnn/functional/model.py`

---

**Document Owner**: Architecture Team  
**Last Review**: 2025-11-03  
**Next Review**: After Step 4 completion
