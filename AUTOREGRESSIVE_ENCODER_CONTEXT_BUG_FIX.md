# Autoregressive Sampling Encoder Context Bug Fix

## Summary

Fixed a **critical bug** in autoregressive sampling where the encoder context was using central node features (`h_i`) instead of neighbor features (`h_j`). This is the **exact same bug** that was previously fixed in the unconditional decoder, but it still existed in the autoregressive sampling path.

**Impact**: This bug was causing ~5% sequence recovery (essentially random) in temperature sampling. The fix should restore performance to 40-60% recovery.

---

## The Bug

### Location
**File**: `src/prxteinmpnn/model/mpnn.py`
**Function**: `_run_autoregressive_scan`
**Lines**: 705-721 (before fix)

### What Was Wrong

The encoder context construction was:

```python
# BEFORE (BUGGY)
encoder_edge_neighbors = concatenate_neighbor_nodes(
  jnp.zeros_like(node_features),
  edge_features,
  neighbor_indices,
)  # Returns [e_ij, 0]

encoder_context = jnp.concatenate(
  [
    jnp.tile(
      jnp.expand_dims(node_features, -2),  # ❌ BUG: h_i (central node)
      [1, edge_features.shape[1], 1],
    ),
    encoder_edge_neighbors,
  ],
  -1,
)  # Creates [h_i, e_ij, 0] - WRONG!

encoder_context = encoder_context * mask_fw[..., None]
```

This created encoder context as `[h_i, e_ij, 0]` where:
- **`h_i`**: The central node's encoder output (same value tiled for ALL neighbors)
- **`e_ij`**: Edge features between node i and neighbor j
- **`0`**: Placeholder for sequence embeddings (zeros for not-yet-decoded positions)

### Why This Is Wrong

In autoregressive decoding, positions can attend to two types of neighbors:

1. **Decoded neighbors** (mask_bw=1): Use full context `[e_ij, s_j, h_j]`
2. **Not-yet-decoded neighbors** (mask_fw=1): Use encoder context `[?, ?, ?]`

The encoder context should provide structural information from neighbors that haven't been decoded yet. But using `h_i` (central node) means **all neighbors provide the same structural information** - the central node's features!

This defeats the purpose of the graph neural network, which should aggregate unique information from each neighbor.

### Concrete Example

Consider position 10 decoding with neighbors [5, 8, 12, 15]:
- Neighbor 5: already decoded (mask_bw=1) → use `[e_10,5, s_5, h_5]` ✓
- Neighbor 8: already decoded (mask_bw=1) → use `[e_10,8, s_8, h_8]` ✓
- Neighbor 12: not yet decoded (mask_fw=1) → use encoder context
- Neighbor 15: not yet decoded (mask_fw=1) → use encoder context

**Before fix**: Encoder context for neighbors 12 and 15 was `[h_10, e_10,12/15, 0]`
→ Both neighbors provide `h_10` (central node features) - no spatial variation!

**After fix**: Encoder context for neighbors 12 and 15 is `[e_10,12/15, 0, h_12/15]`
→ Each neighbor provides its own unique encoder features ✓

---

## The Fix

### Code Changes

```python
# AFTER (FIXED)
encoder_edge_neighbors = concatenate_neighbor_nodes(
  jnp.zeros_like(node_features),
  edge_features,
  neighbor_indices,
)  # Returns [e_ij, 0]

# Gather neighbor encoder features h_j (not central node h_i!)
encoder_context = concatenate_neighbor_nodes(
  node_features,  # ✓ This will be gathered as h_j = node_features[neighbors]
  encoder_edge_neighbors,  # [e_ij, 0]
  neighbor_indices,
)  # Returns [[e_ij, 0], h_j] = [e_ij, 0, h_j]

encoder_context = encoder_context * mask_fw[..., None]
```

Now the encoder context is `[e_ij, 0, h_j]` where:
- **`e_ij`**: Edge features (same as before)
- **`0`**: Placeholder for sequence embeddings (same as before)
- **`h_j`**: **Neighbor's encoder features** (FIXED!)

Each neighbor now provides its own unique structural information via `h_j`.

---

## Reference Implementation

**ColabDesign** (`colabdesign/mpnn/sample.py` lines 50-52):

```python
h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)  # Gathers h_V[neighbors] = h_j
h_EXV_encoder = mask_fw[...,None] * h_EXV_encoder
```

ColabDesign explicitly gathers `h_V[neighbors]` (neighbor encoder features), confirming this is the correct approach.

---

## Why This Matters

### Without the fix:
- **Temperature sampling recovery**: ~5% (random guessing)
- **Root cause**: Model can't distinguish between different neighbor contexts when using encoder information
- **Symptom**: All predicted sequences have similar poor quality regardless of structural variations

### With the fix:
- **Expected temperature sampling recovery**: 40-60%
- **Improvement**: Model can now properly aggregate structural information from each neighbor
- **Result**: Sequences should match the protein's structure-function relationship

---

## Relationship to Previous Fixes

This is the **THIRD instance** of the "using `h_i` instead of `h_j`" bug:

1. **First instance (fixed)**: Unconditional decoder
   - **File**: `src/prxteinmpnn/model/decoder.py`
   - **Symptom**: 64.5% Alanine bias in unconditional predictions
   - **Fix**: Changed decoder to gather `h_j` instead of tiling `h_i`

2. **Second instance (fixed earlier in this session)**: Conditional decoder
   - **File**: `src/prxteinmpnn/model/decoder.py` - `call_conditional()`
   - **Symptom**: ~30% sequence recovery instead of >90%
   - **Fix**: Removed incorrect `attention_mask` parameter (which was related to h_i/h_j confusion)

3. **Third instance (this fix)**: Autoregressive sampling
   - **File**: `src/prxteinmpnn/model/mpnn.py` - `_run_autoregressive_scan()`
   - **Symptom**: ~5% sequence recovery in temperature sampling
   - **Fix**: Changed encoder context to gather `h_j` instead of tiling `h_i`

The pattern is clear: **whenever constructing context for graph message passing, we must gather features from neighbors (`h_j`), not tile features from the central node (`h_i`).**

---

## Testing

To verify this fix:

```bash
# Run temperature sampling test
uv run pytest tests/sampling/test_sequence_recovery.py::test_temperature_sampling_recovery_1ubq -v

# Expected result:
# Mean recovery: 40-60% (up from ~5%)
```

---

## Commit

```
fix: critical bug in autoregressive sampling encoder context

Fixed the encoder context construction in autoregressive sampling to use
neighbor encoder features (h_j) instead of central node features (h_i).

This is the SAME bug that was already fixed in the unconditional decoder.

Files modified:
- src/prxteinmpnn/model/mpnn.py
```

---

## Date

2025-11-07
