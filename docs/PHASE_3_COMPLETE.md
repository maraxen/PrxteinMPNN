# Phase 3 Complete: Sampling Adapter Functions

**Date:** 2025-01-XX  
**Status:** ✅ Complete

## Overview

Created adapter functions in `src/prxteinmpnn/sampling/adapter.py` to provide a unified interface for working with both legacy PyTree and new Equinox PrxteinMPNN models. This enables gradual migration without breaking existing code.

## Changes Made

### 1. Created `src/prxteinmpnn/sampling/adapter.py`

Implemented four key adapter functions:

#### `is_equinox_model(model: Model) -> bool`
- Detects whether a model is a PrxteinMPNN Equinox instance or legacy PyTree
- Uses `isinstance()` check with runtime import to avoid circular dependencies
- Returns `True` for Equinox models, `False` for PyTree

#### `get_encoder_fn(model: Model, ...) -> Callable`
- Returns an encoder function that works with either architecture
- For Equinox models: Returns a wrapper around `model.encoder`
- For PyTree models: Calls `make_encoder()` from functional module
- Accepts optional parameters: `attention_mask_type`, `num_encoder_layers`, `scale`

#### `get_decoder_fn(model: Model, ...) -> Callable`
- Returns a decoder function that works with either architecture
- For Equinox models: Returns `model.decoder` based on decoding approach
- For PyTree models: Calls `make_decoder()` from functional module
- Accepts optional parameters: `attention_mask_type`, `decoding_approach`, `num_decoder_layers`

#### `get_model_parameters(model: Model) -> ModelParameters`
- Extracts ModelParameters PyTree from either architecture
- For PyTree models: Returns the model directly (it IS the parameters)
- For Equinox models: Raises `NotImplementedError` (not needed in new architecture)

### 2. Type Safety

- Imported proper Literal types:
  - `MaskedAttentionType` from `prxteinmpnn.model.masked_attention`
  - `DecodingApproach` from `prxteinmpnn.model.decoder`
- All type annotations satisfy pyright strict mode
- All code passes ruff linting with strict rules

### 3. Created Comprehensive Tests

**File:** `tests/sampling/test_adapter.py`

- 12 tests total, all passing ✅
- Tests cover:
  - Model type detection (`is_equinox_model`)
  - Encoder function creation for Equinox models
  - Decoder function creation for Equinox models
  - Parameter extraction (raises NotImplementedError for Equinox)
  - All 8 model combinations (4 versions × 2 weights)

**Test Results:**
```
12 passed in 9.80s
```

**Note:** Tests only cover Equinox architecture since legacy `.pkl` files no longer exist on HuggingFace. The adapter functions are designed to support both architectures, but in practice we're migrating to Equinox.

## Code Quality

### Linting (Ruff)
```bash
$ ruff check src/prxteinmpnn/sampling/adapter.py
All checks passed!
```

### Type Checking (Pyright)
```bash
$ pyright src/prxteinmpnn/sampling/adapter.py
0 errors, 0 warnings, 0 informations
```

## Next Steps (Phase 4)

According to `docs/MIGRATION_QUICK_START.md`, the next phase is:

**Phase 4: Update Sampling & Scoring**

1. Update `src/prxteinmpnn/sampling/sampling.py`:
   - Replace direct calls to `make_encoder` and `make_decoder`
   - Use `get_encoder_fn()` and `get_decoder_fn()` from adapter
   - Accept `Model` union type instead of `ModelParameters`

2. Update `src/prxteinmpnn/scoring/scoring.py`:
   - Similar changes to sampling.py
   - Use adapter functions for encoder/decoder creation

3. Create tests:
   - Update existing tests to work with both architectures
   - Add parametrized tests (`@pytest.mark.parametrize`)
   - Verify numerical equivalence where possible

## Migration Progress

- ✅ Phase 1: Adapter Layer (`functional/model.py`)
- ✅ Phase 2: Run Utilities (`run/prep.py`, type system)
- ✅ Phase 3: Sampling Adapter Functions (`sampling/adapter.py`)
- ⏳ Phase 4: Update Sampling & Scoring (next)
- ⏳ Phase 5: Flip the Switch (default to new architecture)
- ⏳ Phase 6: Clean Up Legacy Code

## Files Modified

1. **Created:**
   - `src/prxteinmpnn/sampling/adapter.py` (170 lines)
   - `tests/sampling/test_adapter.py` (61 lines)

2. **No breaking changes** - all existing code continues to work

## Validation

- ✅ All adapter functions work correctly
- ✅ Type checking passes (pyright strict mode)
- ✅ Linting passes (ruff strict rules)
- ✅ All 12 tests passing
- ✅ All 8 model versions load and create encoder/decoder functions
- ✅ No circular import issues

## Summary

Phase 3 successfully creates the adapter layer for sampling functions. The `sampling/adapter.py` module provides a clean interface that:

1. Detects model architecture type
2. Returns appropriate encoder/decoder functions
3. Handles type safety with proper Literal types
4. Supports all model versions and weights
5. Enables gradual migration to Equinox architecture

This completes the foundational adapter infrastructure. The next phase will integrate these adapters into the actual sampling and scoring code.
