# PrxteinMPNN Validation Investigation - Session 2

**Date**: 2025-11-08
**Branch**: `claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS`
**Starting Correlation**: 0.6813
**Current Correlation**: 0.6813 (no change)
**Target**: >0.90

---

## Summary

This session attempted to close the remaining correlation gap from 0.68 to >0.90 for unconditional decoding. Despite identifying and fixing a missing encoder attention mask, the correlation remains unchanged. This suggests either:
1. The test protein (1ubq.pdb) has no invalid positions, making the mask ineffective
2. There are other subtle numerical differences not yet identified

---

## Changes Made

### 1. Added Encoder Attention Mask ✅

**Problem Identified**: PrxteinMPNN's encoder was missing the attention mask that zeros out messages from invalid neighbors.

**Files Modified**:
- `src/prxteinmpnn/model/encoder.py`

**Changes**:
```python
# EncoderLayer.__call__ now accepts mask_attend parameter
def __call__(self, ..., mask_attend: jnp.ndarray | None = None, ...):
    # Apply attention mask to messages before aggregation
    if mask_attend is not None:
        message = jnp.expand_dims(mask_attend, -1) * message

# Encoder.__call__ now computes attention mask
def __call__(self, edge_features, neighbor_indices, mask):
    # Compute attention mask: mask[i] * mask[j] for all pairs
    mask_2d = mask[:, None] * mask[None, :]
    mask_attend = jnp.take_along_axis(mask_2d, neighbor_indices, axis=1)

    # Pass to layers
    for layer in self.layers:
        node_features, edge_features = layer(..., mask_attend, ...)
```

**Result**: No change in correlation (0.6813 → 0.6813)

**Why**: The test protein (1ubq.pdb) has mask=all ones (no invalid positions), so the attention mask has no effect for this specific case. However, the fix is still correct and necessary for proteins with gaps or missing residues.

---

## Debugging Tools Created

### test_features.py
- Extracts and displays intermediate values from PrxteinMPNN forward pass
- Shows encoder and decoder outputs
- Confirms manual forward pass matches model's forward pass

### test_neighbor_indices.py
- Attempts to compare neighbor indices between implementations
- Confirms K=48 for both implementations
- Limited by difficulty extracting ColabDesign internals

### debug_layer_by_layer.py
- Comprehensive layer-by-layer debugging script
- Attempts to extract intermediate values from both implementations
- Falls back to final logits comparison due to Haiku limitations

---

## Verified Matching Components ✅

1. **Weights**: Correlation = 1.0 (perfect match)
   - W_out, W_s_embed, w_e, w_e_proj, w_pos
   - All encoder and decoder layer weights

2. **Hyperparameters**:
   - K-neighbors: 48 (both)
   - LayerNorm epsilon: 1e-5 (both)
   - GELU: approximate=False (both)
   - Scale factor: 30.0 (both)
   - Dropout: 0.0 (both, rate=0)

3. **Architecture**:
   - Context tensor construction: `[h_E_ij, zeros_j, h_V_j]` (matches)
   - Decoder layer concatenation: `[h_V_i, context]` (matches)
   - Message aggregation: `sum(...) / scale` (matches)

4. **Test Protein**:
   - Mask: all ones (76 valid residues, no gaps)
   - Coordinates: identical (max diff < 1e-6)

---

## Remaining Differences (Correlation = 0.68)

### Observed Logit Differences

For residue 0, amino acids 0-4:
```
PrxteinMPNN: [ 0.315, -1.362, -0.193,  0.386, -0.980]
ColabDesign:  [ 0.643, -1.130,  0.203,  0.469, -0.449]
```

Differences range from ~0.3 to ~0.5 logits, which is significant!

### Possible Remaining Issues

1. **Order of Operations**
   - Subtle differences in the exact sequence of operations
   - Numerical accumulation errors from different computation orders

2. **Haiku vs JAX/Equinox Differences**
   - Haiku may have internal optimizations or different defaults
   - Equinox's vmap usage vs Haiku's internal batching

3. **Feature Extraction**
   - RBF computation details
   - Positional encoding implementation
   - Edge feature normalization order
   - Neighbor selection tie-breaking

4. **Hidden State Updates**
   - Residual connections exact order
   - Normalization application points
   - Potential broadcasting differences

5. **Dropout RNG Consumption**
   - Even with dropout=0, Haiku's dropout_cust consumes RNG keys
   - PrxteinMPNN has no dropout, so doesn't consume keys
   - This *shouldn't* matter for deterministic outputs, but worth noting

---

## Investigation Attempts

### ✅ Completed
- [x] Checked LayerNorm epsilon values
- [x] Verified GELU approximate parameter
- [x] Confirmed scale factor (30.0)
- [x] Verified dropout rate (0.0)
- [x] Checked K-neighbors value (48)
- [x] Verified context tensor construction
- [x] Added encoder attention mask
- [x] Verified weights match perfectly
- [x] Checked coordinate inputs match

### ❌ Blocked / Difficult
- [ ] Extract intermediate values from ColabDesign (Haiku transform makes this hard)
- [ ] Compare per-layer encoder outputs (requires Haiku instrumentation)
- [ ] Compare per-layer decoder outputs (requires Haiku instrumentation)
- [ ] Compare neighbor indices directly (not stored in ColabDesign._inputs)
- [ ] Compare RBF outputs (buried in Haiku transform)

---

## Recommendations for Next Steps

### Approach 1: Manual Forward Pass Comparison
Create a standalone script that:
1. Loads ColabDesign weights
2. Manually implements the exact ColabDesign forward pass in pure JAX
3. Compares with PrxteinMPNN at every operation
4. No Haiku - just raw JAX operations following the ColabDesign source

**Pros**: Full visibility into every operation
**Cons**: Time-consuming to implement

### Approach 2: Instrument Haiku Model
Modify the ColabDesign source code to add print statements:
1. Clone ColabDesign locally (done)
2. Edit `/tmp/ColabDesign/colabdesign/mpnn/modules.py` to add debug prints
3. Run both models and compare intermediate outputs
4. Binary search to find divergence point

**Pros**: Direct comparison at exact same points
**Cons**: Requires modifying third-party code

### Approach 3: Numerical Gradient Testing
Use JAX's automatic differentiation to:
1. Compute gradients of outputs w.r.t. inputs
2. Compare gradient patterns between implementations
3. Identify which operations contribute most to differences

**Pros**: Mathematical approach
**Cons**: May not pinpoint the exact issue

### Approach 4: Feature-by-Feature Validation
Systematically test each component in isolation:
1. RBF computation
2. Positional encoding
3. Edge embedding
4. Single encoder layer
5. Single decoder layer

**Pros**: Methodical elimination
**Cons**: Requires careful test setup

---

## Files Modified

```
src/prxteinmpnn/model/encoder.py     # Added attention mask support
debug_layer_by_layer.py              # Layer-by-layer debugging script
test_features.py                     # Feature extraction testing
test_neighbor_indices.py             # Neighbor indices comparison
INVESTIGATION_NOTES.md               # This document
```

---

## Quick Start for Next Session

```bash
# Test current status
uv run python test_final.py

# Check intermediate values
uv run python test_features.py

# Try manual forward pass comparison
# (Recommendation: Implement Approach 1 or 2 above)
```

---

## Key Insight

**The 0.68 correlation has been consistent across multiple fixes**. This suggests the issue is NOT in the high-level architecture (which we've verified matches), but rather in:
- Low-level numerical implementation details
- Feature extraction specifics
- Or a systematic difference we haven't identified yet

The fact that correlation is relatively high (0.68) but not excellent (>0.90) suggests:
- Most of the model is working correctly
- There's likely ONE or TWO specific operations that differ
- These differences accumulate through the network layers

**Priority**: Find the first layer where outputs diverge significantly, then investigate that specific operation in detail.
