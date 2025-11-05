# Blocked Modules - TODO List

This document tracks modules that are temporarily disabled during the Equinox migration and require refactoring work before they can be re-enabled.

## Status Summary

- **Total Pyright Errors**: 3 (down from 31)
- **Blocked Modules**: 2
- **Reason**: Depend on deleted `conditional_logits` and `unconditional_logits` modules

## Blocked Modules

### 1. `src/prxteinmpnn/run/jacobian.py`

**Status**: ❌ Disabled - Imports missing modules

**Errors**:
```
src/prxteinmpnn/run/jacobian.py:35:6 - error: Import "prxteinmpnn.sampling.conditional_logits" could not be resolved
```

**Dependencies**:
- `prxteinmpnn.sampling.conditional_logits.ConditionalLogitsFn` (deleted)
- `prxteinmpnn.sampling.conditional_logits.make_conditional_logits_fn` (deleted)
- `prxteinmpnn.sampling.conditional_logits.make_encoding_conditional_logits_split_fn` (deleted)

**Required Refactoring**:
1. Create new `conditional_logits` module that uses `PrxteinMPNN` model with `decoding_approach="conditional"`
2. Implement factory functions:
   - `make_conditional_logits_fn(model: PrxteinMPNN) -> ConditionalLogitsFn`
   - `make_encoding_conditional_logits_split_fn(model: PrxteinMPNN) -> ...`
3. Update jacobian.py to use new API

**Notes**:
- The Equinox `PrxteinMPNN` model already supports conditional mode via `jax.lax.switch`
- Need to extract and expose the conditional decoding branch as a standalone function
- Jacobian computation should use the conditional logits function for computing sensitivities

### 2. `src/prxteinmpnn/run/conformational_inference.py`

**Status**: ❌ Disabled - Imports missing modules

**Errors**:
```
src/prxteinmpnn/run/conformational_inference.py:24:6 - error: Import "prxteinmpnn.sampling.conditional_logits" could not be resolved
src/prxteinmpnn/run/conformational_inference.py:25:6 - error: Import "prxteinmpnn.sampling.unconditional_logits" could not be resolved
```

**Dependencies**:
- `prxteinmpnn.sampling.conditional_logits.make_conditional_logits_fn` (deleted)
- `prxteinmpnn.sampling.unconditional_logits.make_unconditional_logits_fn` (deleted)

**Required Refactoring**:
1. Create new `conditional_logits` module (same as jacobian.py requirement)
2. Create new `unconditional_logits` module that uses `PrxteinMPNN` model with `decoding_approach="unconditional"`
3. Implement factory functions:
   - `make_conditional_logits_fn(model: PrxteinMPNN) -> ConditionalLogitsFn`
   - `make_unconditional_logits_fn(model: PrxteinMPNN) -> UnconditionalLogitsFn`
4. Update conformational_inference.py to use new API

**Notes**:
- Conformational inference needs both conditional and unconditional logits for state inference
- The model already supports both modes via `jax.lax.switch`
- Need to ensure batch processing and streaming work correctly with new API

## Refactoring Plan

### Phase 1: Create `conditional_logits` module

1. Create `src/prxteinmpnn/sampling/conditional_logits.py`
2. Define `ConditionalLogitsFn` protocol/type alias
3. Implement `make_conditional_logits_fn(model: PrxteinMPNN) -> ConditionalLogitsFn`
4. Implement `make_encoding_conditional_logits_split_fn(model: PrxteinMPNN)` (for jacobian)
5. Add tests for conditional logits functionality

### Phase 2: Create `unconditional_logits` module

1. Create `src/prxteinmpnn/sampling/unconditional_logits.py`
2. Define `UnconditionalLogitsFn` protocol/type alias
3. Implement `make_unconditional_logits_fn(model: PrxteinMPNN) -> UnconditionalLogitsFn`
4. Add tests for unconditional logits functionality

### Phase 3: Update blocked modules

1. Update `jacobian.py` to import and use new conditional logits API
2. Update `conformational_inference.py` to import and use both conditional and unconditional logits APIs
3. Re-enable imports in `run/__init__.py`
4. Update tests

### Phase 4: Validation

1. Run Pyright to verify no remaining type errors
2. Run test suite to ensure all functionality works
3. Test jacobian computation on sample proteins
4. Test conformational inference on ensemble data
5. Update documentation

## Technical Notes

### Model Architecture Compatibility

The `PrxteinMPNN` model uses `jax.lax.switch` to dispatch between three decoding modes:
- Mode 0: Unconditional (no sequence input)
- Mode 1: Conditional (sequence input provided)
- Mode 2: Autoregressive (sequential decoding with sampling)

To create logits functions, we need to:
1. Extract the appropriate branch from the model
2. Wrap it in a function with the expected signature
3. Ensure it's compatible with JAX transformations (jit, vmap, etc.)

### Signature Compatibility

All branches accept the same parameters due to `jax.lax.switch` requirements:
```python
def _call_conditional(
    self,
    coords: Coordinates,
    S: ProteinSequence | None,
    chain_mask: ChainMask,
    chain_encoding_all: Encodings,
    residue_idx: ResidueIndex,
    mask: Mask,
    chain_M_pos: ChainMask,
    omit_AAs_np: OmitAAs,
    bias: Logits | None,
    randoms: RandomSamples,
    S_true: ProteinSequence,
    temperature: float,
) -> tuple[ProteinSequence, Logits]:
```

The logits functions should simplify this signature to just the essential parameters needed for inference.

## Timeline

- **Target Completion**: After testing suite updates
- **Priority**: Medium (blocks advanced features but core functionality works)
- **Dependencies**: None (can be done after testing suite work)

## Related Files

- `src/prxteinmpnn/model/mpnn.py` - Contains the three decoding modes
- `src/prxteinmpnn/sampling/sample.py` - Example of using autoregressive mode
- `tests/test_mpnn.py` - Tests for model modes
- `MIGRATION_COMPLETE.md` - Overall migration status

## References

- Original conditional_logits module (deleted): Used functional model with static call
- Original unconditional_logits module (deleted): Used functional model with static call
- New Equinox architecture: Uses object-oriented `PrxteinMPNN` class with mode switching

## Notes for Future Implementation

When implementing the new logits modules:

1. **Maintain JAX Compatibility**: Ensure all functions are compatible with `jax.jit`, `jax.vmap`, and `jax.scan`
2. **Use Type Hints**: Follow strict Pyright typing with proper type annotations
3. **Follow Project Style**: Use Google-style docstrings and adhere to Ruff linting rules
4. **Test Thoroughly**: Include unit tests for each logits function and integration tests with blocked modules
5. **Document Well**: Update this document as work progresses and add examples to docstrings
