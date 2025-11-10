# PrxteinMPNN ColabDesign Equivalence Validation

## Summary

The PrxteinMPNN implementation has been fully validated against the original ColabDesign ProteinMPNN implementation. All three decoding paths achieve >0.95 correlation with ColabDesign! ✅

## Test Results

| Path | Correlation | Status | Target |
|------|------------|--------|---------|
| **Unconditional** | 0.984 | ✅ PASS | >0.95 |
| **Conditional** | 0.958-0.984 | ✅ PASS | >0.95 |
| **Autoregressive** | 0.953-0.970 | ✅ PASS | >0.95 |

## Continuous Integration

The equivalence tests are part of the test suite and can be run with:

```bash
pytest tests/model/test_colabdesign_equivalence.py -v
```

**Prerequisites**: ColabDesign must be installed:
```bash
pip install git+https://github.com/sokrypton/ColabDesign.git@e31a56f
```

Test suite includes:
- `test_unconditional_logits`: Validates structure-based predictions
- `test_conditional_logits`: Validates fixed sequence scoring
- `test_autoregressive_sampling`: Validates sequential generation
- `test_ar_first_step_matches_unconditional`: Sanity check for AR implementation
- `test_conditional_with_zero_mask_matches_unconditional`: Sanity check for conditional implementation

## Bugs Fixed

### 1. Atom Ordering (0.681 → 0.984)
**Problem**: Parser outputs PDB order (O at index 3, CB at 4) but code used atom37 indices (CB at 3, O at 4)

**Solution**: Created `atom_ordering.py` with proper PDB_ORDER_INDICES constants

**Files Modified**:
- Created: `src/prxteinmpnn/utils/atom_ordering.py`
- Modified: `src/prxteinmpnn/utils/coordinates.py`

### 2. Double Masking in Conditional Decoder (0.872 → 0.984)
**Problem**: Conditional decoder was applying attention masking TWICE:
1. Once when constructing `layer_edge_features` (correct)
2. Again inside the decoder layer by multiplying messages by `attention_mask` (incorrect)

When `ar_mask=0`, this caused all messages to be zeroed out, preventing decoder updates.

**Solution**: Removed the `attention_mask` parameter from layer calls in conditional decoder (line 354-359 in `decoder.py`)

**Key Insight**: ColabDesign only masks the edge features, not the messages. The masking is already encoded in the edge features passed to each layer.

### 3. Wrong Encoder Context in Autoregressive Path (0.218 → 0.970)
**Problem**: Autoregressive sampling was constructing encoder context as `[h_i, e_ij, 0_j]` instead of `[e_ij, 0_j, h_j]`

This caused the first AR step (position 0 with no context) to produce completely different logits from unconditional (diff 5.25).

**Solution**: Changed encoder context construction to use `concatenate_neighbor_nodes` properly (lines 708-720 in `mpnn.py`)

**Before**:
```python
encoder_context = jnp.concatenate([
  jnp.tile(node_features, ...),  # h_i tiled
  encoder_edge_neighbors,         # [e_ij, 0_j]
], -1)  # Result: [h_i, e_ij, 0_j] ❌
```

**After**:
```python
encoder_context = concatenate_neighbor_nodes(
  node_features,
  encoder_edge_neighbors,
  neighbor_indices,
)  # Result: [[e_ij, 0_j], h_j] = [e_ij, 0_j, h_j] ✅
```

## Validation Tests

- `test_fix_correlation.py` - Original validation (unconditional)
- `test_all_paths_simplified.py` - All three paths
- `test_conditional_autoregressive.py` - Focused conditional & AR testing
- `test_conditional_fixed_ar_mask.py` - Conditional with explicit ar_mask=0
- `debug_conditional_detailed.py` - Found double masking bug
- `debug_autoregressive.py` - AR mask construction testing
- `debug_ar_first_step.py` - Found encoder context bug

## Technical Details

### Alphabet Conversion
- AlphaFold: `ARNDCQEGHILKMFPSTWYVX`
- MPNN: `ACDEFGHIKLMNPQRSTVWYX`

### Decoding Approaches

1. **Unconditional**: Pure structure-based prediction, no sequence input
   - Context: `[e_ij, 0_j, h_j]` (constant through all layers)

2. **Conditional**: Fixed sequence scoring with autoregressive masking
   - Context: `mask_bw * [e_ij, s_j, h_j] + mask_fw * [e_ij, 0_j, h_j]`
   - When `ar_mask=0`: reduces to unconditional

3. **Autoregressive**: Sequential generation with Gumbel-max sampling
   - Encoder context: `[e_ij, 0_j, h_j]` (from encoder, masked by `mask_fw`)
   - Decoder context: `[e_ij, s_j, h_j]` (updated each step, masked by `mask_bw`)

## Conclusion

The PrxteinMPNN implementation now correctly replicates ColabDesign's behavior across all three decoding paths, with correlations >0.95 for all paths. The core architecture is validated! 🎉
