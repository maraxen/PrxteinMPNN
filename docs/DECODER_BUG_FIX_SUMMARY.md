# Decoder Bug Fix: Complete Technical Summary

**Date:** November 7, 2025  
**Status:** ✅ CRITICAL BUG FIXED - Alanine Bias Resolved  
**Impact:** Reduced Alanine predictions from 64.5% to 3.9%

---

## Executive Summary

A critical architectural bug was discovered and fixed in the PrxteinMPNN unconditional decoder. The decoder was incorrectly using **central node features (h_i)** instead of **neighbor node features (h_j)** when constructing the edge context for message passing. This caused systematic bias toward Alanine predictions.

### Key Metrics

- **Before Fix:** 64.5% Alanine predictions (unconditional mode)
- **After Fix:** 3.9% Alanine predictions (unconditional mode)
- **Decoder Feature Alignment:** Was 4.3× biased toward Alanine, now corrected

---

## Root Cause Analysis

### The Bug

The unconditional decoder was constructing layer edge features as:

```python
# INCORRECT (what we had)
layer_edge_features = [h_i, zeros, e_ij]  # shape: (L, K, 384)
# where h_i = central node features (repeated for all neighbors)
```

Should have been:

```python
# CORRECT (what we need)
layer_edge_features = [e_ij, zeros, h_j]  # shape: (L, K, 384)
# where h_j = neighbor node features (gathered per edge)
```

### Why This Caused Alanine Bias

1. **Missing Neighbor Information:** By using h_i (central node) instead of h_j (neighbors), the decoder received no information about neighboring residues
2. **Collapsed Context:** All edges from a node received identical node features, eliminating spatial context
3. **Feature Misalignment:** The resulting decoded features became systematically aligned with Alanine's weight vector
4. **Diagnostic Evidence:** `decoded_features · alanine_weights = +1.051` vs `decoded_features · other_weights = -0.244` (4.3× ratio)

---

## Debugging Journey: What Was Validated

### ✅ Confirmed Working Correctly

1. **Weight Structure** (60 modules total)
   - All encoder/decoder layers present
   - All weight matrices correctly loaded from original checkpoint
   - Verified via `tests/debug/test_weight_values.py`
   - Documented in `docs/WEIGHT_VERIFICATION.md`

2. **Alphabet Conversion**
   - MPNN alphabet order correctly implemented
   - Conversion between indices working properly
   - No off-by-one errors

3. **Bias Application**
   - Output layer bias (+0.1816 for Alanine) applied correctly
   - Bias is NOT the cause of the 64.5% predictions
   - Alanine has second-LOWEST weight norm (1.854), not highest

4. **Coordinate Processing**
   - No NaN or Inf values in coordinates
   - Edge distances reasonable (mean ~5-15 Å)
   - RBF encoding functioning correctly

5. **Feature Extraction**
   - Edge features healthy (mean=0.171, std=4.389)
   - No collapsed or degenerate features
   - Neighbor gathering working correctly

6. **Encoder**
   - Node features after encoder are healthy
   - Mean=0.032, std=0.367, CV=0.0186 (good variation)
   - No evidence of collapsed representations

7. **Output Projection**
   - Weight matrix and bias correctly applied
   - Logits computation: `W·h + b` verified correct
   - Problem was in the input to this layer (decoder output)

### ❌ Identified as Buggy (Now Fixed)

**Unconditional Decoder** - Using wrong node features in edge context construction

---

## The Fix: Technical Details

### Files Modified

#### 1. `src/prxteinmpnn/model/decoder.py` (lines 209-275)

**Before:**

```python
def __call__(
  self,
  node_features: Float[Array, "n_nodes n_features"],
  edge_features: Float[Array, "n_nodes n_neighbors edge_features"],
  mask: Float[Array, "n_nodes"],
) -> Float[Array, "n_nodes n_features"]:
  """Unconditional forward pass (no sequence information)."""
  # BUG: Using central node features (h_i)
  nodes_expanded = jnp.tile(
    jnp.expand_dims(node_features, -2), 
    (1, edge_features.shape[-2], 1)
  )
  zeros_expanded = jnp.zeros_like(nodes_expanded)
  
  # Wrong context: [h_i, zeros, e_ij]
  layer_edge_features = jnp.concatenate(
    [nodes_expanded, zeros_expanded, edge_features], axis=-1
  )
  
  loop_node_features = node_features
  for layer in self.layers:
    loop_node_features = layer(loop_node_features, layer_edge_features, mask)
  
  return loop_node_features
```

**After:**

```python
def __call__(
  self,
  node_features: Float[Array, "n_nodes n_features"],
  edge_features: Float[Array, "n_nodes n_neighbors edge_features"],
  neighbor_indices: NeighborIndices,  # NEW PARAMETER
  mask: Float[Array, "n_nodes"],
) -> Float[Array, "n_nodes n_features"]:
  """Unconditional forward pass (no sequence information)."""
  zeros_expanded = jnp.zeros((node_features.shape[0], neighbor_indices.shape[-1], 128))
  
  loop_node_features = node_features
  for layer in self.layers:
    # FIXED: Gather neighbor features (h_j) at each layer
    edge_and_neighbors = concatenate_neighbor_nodes(
      loop_node_features,
      edge_features,
      neighbor_indices,
    )
    
    # Correct context: [e_ij, zeros, h_j]
    layer_edge_features = jnp.concatenate([
      edge_and_neighbors[..., :128],   # e_ij (edge features)
      zeros_expanded,                   # s_j (sequence embeddings, zeros in unconditional)
      edge_and_neighbors[..., 128:],    # h_j (NEIGHBOR node features)
    ], axis=-1)
    
    loop_node_features = layer(loop_node_features, layer_edge_features, mask)
  
  return loop_node_features
```

**Key Changes:**

1. Added `neighbor_indices` parameter to decoder signature
2. Removed static context construction before loop
3. Added per-layer gathering of neighbor features using `concatenate_neighbor_nodes()`
4. Proper context construction: `[e_ij, zeros, h_j]` instead of `[h_i, zeros, e_ij]`
5. Context now updates each layer as `loop_node_features` evolves

#### 2. `src/prxteinmpnn/model/mpnn.py` (line 193)

**Before:**

```python
decoded_node_features = self.decoder(
  node_features,
  processed_edge_features,
  mask,
)
```

**After:**

```python
decoded_node_features = self.decoder(
  node_features,
  processed_edge_features,
  neighbor_indices,  # NEW ARGUMENT
  mask,
)
```

---

## Verification Tests

### Test Results: Before vs After

| Test | Before Fix | After Fix | Status |
|------|-----------|-----------|--------|
| Alanine predictions (unconditional) | 64.5% | 3.9% | ✅ FIXED |
| Prediction diversity | Very low | High | ✅ FIXED |
| Decoded feature alignment | 4.3× Alanine bias | Balanced | ✅ FIXED |
| Intermediate activations test | FAILED | PASSED | ✅ FIXED |
| All sampling tests | PASSED | PASSED | ✅ MAINTAINED |

### Updated Tests

**`tests/debug/test_intermediate_activations.py`** (line 107)

- Updated to pass `neighbor_indices` to decoder
- Now correctly validates the fixed decoder behavior

---

## Current Status: What Works

### ✅ Fully Functional

1. **Tied-position sampling** - All tests passing
2. **Autoregressive masking** - Correctly prevents future positions
3. **Split sampling** - Properly handles tied positions
4. **Temperature sampling** - Diverse outputs at T=2.0
5. **Deterministic sampling** - Reproducible at T=0.1
6. **Unconditional mode** - No longer biased toward Alanine

### ⚠️ Known Limitations

1. **Sequence Recovery:** Currently ~5.7% (expected 40-60%)
   - This is a separate issue from the Alanine bias
   - Likely requires investigation of:
     - Conditional decoder (not yet tested)
     - Sequence context incorporation
     - Temperature/sampling strategy
     - Potential additional architectural differences from reference

---

## Key Diagnostic Tools Created

1. **`tests/debug/test_intermediate_activations.py`**
   - Tracks activations through entire pipeline
   - Computes feature·weight dot products
   - Identifies bias in decoder output
   - **Use this test to verify decoder behavior**

2. **`tests/debug/test_weight_values.py`**
   - Analyzes weight statistics and norms
   - Confirms weight structure integrity

3. **`/tmp/check_decoder_context.py`** (verification script)
   - Validated the bug before fixing
   - Confirmed `concatenate_neighbor_nodes` works correctly

4. **`docs/ALANINE_BIAS_ROOT_CAUSE.md`**
   - Complete analysis of the debugging process

5. **`docs/WEIGHT_VERIFICATION.md`**
   - Confirms all 60 weight modules present

---

## Architecture Reference: ProteinMPNN Decoder

### Correct Edge Feature Context (384-dim)

For unconditional decoding, the edge context should be:

```
[e_ij, s_j, h_j]
```

Where:

- **e_ij** (128-dim): Edge features between nodes i and j
- **s_j** (128-dim): Sequence embedding of neighbor j (zeros in unconditional mode)
- **h_j** (128-dim): Node features of neighbor j (NOT central node i!)

### Why h_j (not h_i) Matters

The decoder performs **message passing from neighbors to the central node**:

- Each edge (i→j) needs information about neighbor j
- Using h_i (central node) prevents information flow from neighbors
- This is fundamental to graph neural network architectures
- The encoder-decoder structure requires different node features at each layer

### Per-Layer Gathering

The fix includes per-layer gathering because:

1. `loop_node_features` evolves through decoder layers
2. Neighbor features (h_j) change at each layer
3. Context must use updated neighbor features, not initial ones
4. This matches the reference ProteinMPNN implementation

---

## Next Steps for Future Investigation

### 1. Sequence Recovery Optimization (~5.7% → 40-60%)

**Potential Issues to Investigate:**

- Conditional decoder implementation (not yet debugged)
- Sequence context integration in conditional mode
- Edge sequence features construction
- Temperature scheduling and sampling strategies
- Comparison with reference implementation logits

**Diagnostic Approach:**

1. Run same structure through ColabDesign/original ProteinMPNN
2. Compare logits position-by-position
3. Identify first divergence point
4. Focus debugging on that specific component

### 2. Performance Optimization

- JIT compilation of decoder (currently working)
- Vectorization of neighbor gathering
- Memory efficiency for large proteins

### 3. Test Coverage

- Add regression test for Alanine bias
- Conditional decoder tests
- Logit comparison tests with reference

---

## Code Quality Notes

### Passes All Quality Checks

- ✅ Ruff linting (line-length=100, all rules)
- ✅ Pyright strict mode type checking
- ✅ All existing tests maintained
- ✅ Google-style docstrings
- ✅ JAX-compatible (JIT/vmap/scan)

### Important Context for Future Work

1. **Always pass neighbor_indices to decoder** - Required for correct behavior
2. **Per-layer gathering is essential** - Context updates as features evolve
3. **Use intermediate activations test** - Best diagnostic for decoder issues
4. **Feature·weight dot products** - Excellent bias detector
5. **Sequence recovery ≠ Alanine bias** - These are separate issues

---

## Quick Reference: Testing Commands

```bash
# Run intermediate activations diagnostic
uv run pytest tests/debug/test_intermediate_activations.py -v -s

# Run all sampling tests
uv run pytest tests/sampling/ -v

# Check sequence recovery rates
uv run pytest tests/sampling/test_sequence_recovery.py -v -s

# Lint check
ruff check src/ --fix

# Type check
pyright src/
```

---

## Critical Context for Fresh Model

### What NOT to investigate (already validated)

- Weight loading/structure
- Alphabet conversion
- Bias application
- Coordinate processing
- Feature extraction
- Encoder behavior
- Output projection mechanics

### What TO investigate next

- **Conditional decoder** (not yet debugged)
- Sequence recovery optimization
- Comparison with reference implementation
- Edge sequence feature construction in conditional mode

### Key Insight

The bug was architectural, not in the weights or basic operations. The decoder was fundamentally unable to aggregate neighbor information because it was looking at the wrong node features. This is now fixed, but sequence recovery optimization is a separate challenge requiring different investigation approaches.

---

## References

- Original ProteinMPNN: [Dauparas et al. 2022]
- JAX implementation guide: Equinox framework
- Graph neural networks: Message passing requires neighbor features (h_j), not self features (h_i)

**End of Summary**
