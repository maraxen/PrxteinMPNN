# Full Functionality TODO

This document tracks all missing functionality that needs to be implemented to achieve complete feature parity with the original implementation.

## Core User API Status

### ✅ Working (Basic)
1. **`sample()`** with temperature sampling
   - ✓ Temperature-based sampling
   - ✓ Bias support
   - ✓ Backbone noise
   - ✓ Basic autoregressive decoding
   
2. **`score()`** 
   - ✓ Sequence scoring against structures
   - ✓ Batch processing
   - ✓ HDF5 streaming output

### ❌ Missing Features

## 1. Sampling Enhancements

### A. Straight-Through Optimization
**Status**: ❌ Not Implemented  
**Priority**: HIGH  
**Files**: 
- `src/prxteinmpnn/sampling/sample.py`
- `src/prxteinmpnn/sampling/ste_optimize.py` (deleted, needs recreation)

**What's Needed**:
1. Implement iterative optimization using straight-through estimator
2. Support `iterations` and `learning_rate` parameters
3. Use unconditional logits as optimization target
4. Integrate with `make_sample_sequences()` factory

**Technical Details**:
- The straight-through estimator (`utils/ste.py`) already exists
- Need to create optimization loop that:
  1. Initializes logits
  2. For each iteration:
     - Apply STE to get discrete sequence
     - Compute loss vs target (unconditional) logits
     - Update logits via gradient descent
  3. Return optimized sequence

**Dependencies**:
- Requires unconditional logits function (see Section 3)

### B. Fixed Positions
**Status**: ❌ Not Implemented  
**Priority**: MEDIUM  
**Files**: `src/prxteinmpnn/sampling/sample.py`

**What's Needed**:
1. Support `fixed_positions` parameter to keep certain residues unchanged
2. Apply mask during autoregressive sampling
3. Ensure fixed positions are not modified during sampling

**Current State**: Parameter accepted but ignored (see line 92 in sample.py)

### C. Tied Positions
**Status**: ⚠️ Partial - Helper functions exist, not integrated  
**Priority**: MEDIUM  
**Files**: 
- `src/prxteinmpnn/utils/autoregression.py` (has `resolve_tie_groups`)
- `src/prxteinmpnn/sampling/sample.py` (needs integration)

**What's Needed**:
1. Integrate `resolve_tie_groups()` into sampling pipeline
2. Support `tied_positions` modes:
   - `"direct"`: All inputs must be same length
   - `"auto"`: Use structure mappings
   - Custom list of tuples
3. Apply tie groups to decoding order and sampling

**Current State**: Helper function exists but not called from anywhere

### D. Average Encodings
**Status**: ❌ Not Implemented  
**Priority**: LOW  
**Files**: `src/prxteinmpnn/run/sampling.py`

**What's Needed**:
1. Average encoder outputs across multiple noise levels
2. Cache averaged encodings for efficiency
3. Use cached encodings for multiple samples

**Current State**: Raises `NotImplementedError` (line 172)

**Technical Notes**:
- This is an optimization for generating multiple samples
- Requires separating encoder from decoder in model
- May require architecture changes to Equinox model

## 2. Scoring Enhancements

### Status: ✅ Core functionality working

**Possible Enhancements**:
- Add support for custom scoring functions
- Add perplexity calculation
- Add per-position scoring

## 3. Jacobian & Conditional Logits

### A. Conditional Logits Module
**Status**: ❌ Deleted, needs recreation  
**Priority**: HIGH (blocks jacobian)  
**Files**: 
- `src/prxteinmpnn/sampling/conditional_logits.py` (deleted)
- Needs: Factory functions for conditional mode

**What's Needed**:
1. Create `make_conditional_logits_fn(model: PrxteinMPNN) -> ConditionalLogitsFn`
   - Wraps model's conditional mode
   - Returns logits for given sequence
   
2. Create `make_encoding_conditional_logits_split_fn(model: PrxteinMPNN)`
   - Separates encoding from decoding
   - Used for jacobian computation

**Technical Details**:
- Model already supports conditional mode via `decoding_approach="conditional"`
- Just need to extract and expose it as standalone function
- Ensure JAX transformations (jit, vmap) work correctly

### B. Unconditional Logits Module
**Status**: ❌ Deleted, needs recreation  
**Priority**: HIGH (needed for straight-through)  
**Files**: `src/prxteinmpnn/sampling/unconditional_logits.py` (deleted)

**What's Needed**:
1. Create `make_unconditional_logits_fn(model: PrxteinMPNN) -> UnconditionalLogitsFn`
   - Wraps model's unconditional mode
   - Returns logits without sequence input
   - Used as optimization target for straight-through

**Technical Details**:
- Model already supports unconditional mode via `decoding_approach="unconditional"`
- Extract and expose as standalone function

### C. Jacobian Computation
**Status**: ❌ Blocked by conditional_logits  
**Priority**: MEDIUM  
**Files**: `src/prxteinmpnn/run/jacobian.py`

**What's Needed**:
1. Fix imports to use new conditional_logits module
2. Update to use Equinox model instead of functional model
3. Test jacobian computation
4. Re-enable in `run/__init__.py`

**Dependencies**:
- Requires conditional_logits module (Section 3.A)
- Requires unconditional_logits module (Section 3.B)

### D. Conformational Inference
**Status**: ❌ Blocked by logits modules  
**Priority**: LOW  
**Files**: `src/prxteinmpnn/run/conformational_inference.py`

**What's Needed**:
1. Fix imports for conditional and unconditional logits
2. Update to use Equinox model
3. Test state inference

## 4. Testing & Validation

### A. Update Test Suite
**Status**: ❌ Many tests broken  
**Priority**: HIGH  

**What's Needed**:
1. Fix `test_eqx_equivalence.py` - imports deleted `functional` module
2. Update sampling tests for new API
3. Update scoring tests
4. Add tests for new features (tied positions, fixed positions, etc.)
5. Update fixtures to use Equinox model

### B. Integration Tests
**Status**: Needed  
**Priority**: MEDIUM  

**What's Needed**:
1. End-to-end test of full sampling pipeline
2. Test streaming to HDF5
3. Test with real PDB files
4. Performance benchmarks

## Implementation Priority

### Phase 1: Core Missing Features (HIGH)
1. ✅ Fix Pyright errors (DONE - 31 → 3)
2. Create unconditional_logits module
3. Create conditional_logits module  
4. Implement straight-through optimization
5. Implement fixed positions

### Phase 2: Advanced Features (MEDIUM)
1. Integrate tied positions
2. Fix and re-enable jacobian
3. Update test suite
4. Integration tests

### Phase 3: Optimizations (LOW)
1. Average encodings feature
2. Fix conformational inference
3. Performance optimizations
4. Documentation updates

## Technical Architecture Notes

### Model Decoding Modes

The `PrxteinMPNN` model supports three modes via `jax.lax.switch`:

```python
# Mode 0: Unconditional - no sequence input
model(..., decoding_approach="unconditional")

# Mode 1: Conditional - sequence provided
model(..., decoding_approach="conditional", S=sequence)

# Mode 2: Autoregressive - sequential sampling
model(..., decoding_approach="autoregressive", prng_key=key)
```

### Creating Logits Functions

To create standalone logits functions:

1. **Unconditional**: 
   ```python
   def make_unconditional_logits_fn(model):
       def logits_fn(coords, mask, res_idx, chain_idx, ...):
           _, logits = model(..., decoding_approach="unconditional")
           return logits
       return logits_fn
   ```

2. **Conditional**:
   ```python
   def make_conditional_logits_fn(model):
       def logits_fn(coords, mask, res_idx, chain_idx, sequence, ...):
           _, logits = model(..., decoding_approach="conditional", S=sequence)
           return logits
       return logits_fn
   ```

### Straight-Through Optimization Loop

```python
def optimize_sequence(model, coords, mask, ..., iterations, learning_rate):
    # Get target (unconditional) logits
    unconditional_fn = make_unconditional_logits_fn(model)
    target_logits = unconditional_fn(coords, mask, ...)
    
    # Initialize logits to optimize
    logits = jnp.zeros_like(target_logits)
    
    # Optimization loop
    for i in range(iterations):
        loss, grads = jax.value_and_grad(ste_loss)(
            logits, target_logits, mask
        )
        logits = logits - learning_rate * grads
    
    # Get final sequence
    sequence = straight_through_estimator(logits).argmax(axis=-1)
    return sequence, logits
```

## Success Criteria

- [ ] All Pyright errors resolved (currently 3 remaining)
- [ ] `sample()` supports both temperature and straight_through
- [ ] `sample()` supports fixed_positions
- [ ] `sample()` supports tied_positions
- [ ] `score()` fully functional (already working)
- [ ] Jacobian computation working
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Examples working

## Timeline Estimate

- **Phase 1**: 2-3 days
- **Phase 2**: 2-3 days  
- **Phase 3**: 1-2 days
- **Total**: ~1 week for full feature parity
