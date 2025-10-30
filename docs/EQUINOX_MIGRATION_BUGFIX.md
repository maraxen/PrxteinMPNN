# Equinox Migration Bugfix - Duplicate Bias Application

## Issue Summary

During numerical equivalence testing between the Equinox and functional implementations of ProteinMPNN, we discovered a critical bug in the `PrxteinMPNN.__call__()` method that caused large numerical discrepancies (max difference ~1.33) between the two implementations.

## Root Cause

The bug was in the output projection step of the `PrxteinMPNN` forward pass:

```python
# BUGGY CODE (before fix)
return jax.vmap(self.w_out)(node_features) + self.b_out
```

The issue: **The bias was being applied twice!**

1. `self.w_out` is an `equinox.nn.Linear` layer that already includes the bias internally
2. The code then **added `self.b_out` again**, effectively doubling the bias

This resulted in:
- Max difference: 1.3264708518981934
- Mean difference: 0.21906955540180206
- The max difference exactly matched the last element of `b_out`: -1.3264695

## Fix

Removed the redundant bias addition:

```python
# FIXED CODE
return jax.vmap(self.w_out)(node_features)
```

After the fix:
- Max difference: 9.536743e-06
- Mean difference: 1.1959175e-06
- All numerical equivalence tests pass with `rtol=1e-5, atol=1e-5`

## Test Results

All 5 numerical equivalence tests now pass:

1. ✅ `test_encoder_numerical_equivalence` - Encoder outputs match within tolerance
2. ✅ `test_full_model_numerical_equivalence` - Full model outputs match within tolerance
3. ✅ `test_model_save_load_equivalence` - Save/load cycle preserves outputs exactly
4. ✅ `test_model_with_different_batch_sizes` - Model handles various input sizes
5. ✅ `test_model_with_partial_masking` - Model correctly handles masked inputs

## Tolerances

The remaining differences (max ~9.5e-6) are due to:
- Floating point accumulation in float32 operations
- Slight differences in operation ordering between implementations
- Multiple layers of transformations accumulating small errors

These are well within acceptable tolerance for float32 operations and indicate numerical equivalence.

## Documentation Updates

Updated the `PrxteinMPNN` class docstring to clarify:
- `w_out`: Linear projection layer that includes bias
- `b_out`: Stored separately for inspection but not used in forward pass (already in `w_out`)

## Lessons Learned

1. **Variable shadowing can hide bugs**: The `edge_features` parameter was shadowed by the encoder output, but wasn't the actual issue
2. **Systematic debugging is key**: By testing encoder, decoder, and projection separately, we isolated the bug to the projection step
3. **Check for redundant operations**: When wrapping functional code in module interfaces, ensure operations aren't duplicated
4. **Test numerical equivalence**: Comprehensive testing revealed the bug that might have been missed otherwise

## Related Files

- Fixed: `src/prxteinmpnn/eqx.py` (PrxteinMPNN.__call__)
- Updated: `tests/test_eqx_equivalence.py` (tolerance adjustments)
- Created: `scripts/convert_weights.py` (conversion infrastructure)

## Migration Status

**Milestone 5: Equinox Migration** ✅ COMPLETE

- [x] Model conversion to Equinox format
- [x] Numerical equivalence verification
- [x] Bug identification and fix
- [x] All tests passing
