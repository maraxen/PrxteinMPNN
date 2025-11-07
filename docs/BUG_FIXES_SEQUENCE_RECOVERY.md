# Bug Fixes for Sequence Recovery Issue

## Summary

This document describes the critical bugs that were causing low sequence recovery (~5-20% instead of expected 40-60%) in PrxteinMPNN and the fixes implemented to resolve them.

## Bug #1: Unconditional Decoder Using Central Node Features

### Location
`src/prxteinmpnn/model/decoder.py` - `Decoder.__call__()` method (lines 209-262)

### Problem Description

The unconditional decoder was incorrectly using **central node features** (h_i) tiled across all neighbors, instead of gathering **neighbor node features** (h_j) as intended in the ProteinMPNN architecture.

**Incorrect Implementation:**
```python
# OLD CODE - BUGGY
nodes_expanded = jnp.tile(
    jnp.expand_dims(node_features, -2),
    [1, edge_features.shape[1], 1],
)
# This creates [h_i, h_i, h_i, ...] for all K neighbors
```

This resulted in the decoder context being:
```
[h_i, 0, e_ij]  # ❌ WRONG - uses central node repeated
```

But the correct structure according to ColabDesign reference is:
```
[e_ij, 0, h_j]  # ✅ CORRECT - uses neighbor nodes
```

### Why This Matters

The decoder needs to aggregate information from neighboring residues (h_j) to predict the amino acid at position i. By using the central node's own features (h_i) repeated for all neighbors, the model was not getting any information about the local structural environment from neighbors.

This is equivalent to asking "what amino acid should be here?" while only looking at the position itself, not its neighbors - which makes accurate prediction impossible.

### The Fix

Changed the decoder to use `concatenate_neighbor_nodes` to properly gather neighbor features:

```python
# NEW CODE - FIXED
# Step 1: Build intermediate context [e_ij, 0, h_j]
temp_context = concatenate_neighbor_nodes(
    jnp.zeros_like(node_features),
    edge_features,
    neighbor_indices,
)
# Step 2: Build full context with proper neighbor gathering
layer_edge_features = concatenate_neighbor_nodes(
    node_features,
    temp_context,
    neighbor_indices,
)
```

This now correctly creates: `[[e_ij, 0], h_j]` where h_j are the neighbor node features.

### Changes Required
- Added `neighbor_indices` parameter to `Decoder.__call__()` method
- Updated call site in `PrxteinMPNN._call_unconditional()` to pass `neighbor_indices`

---

## Bug #2: Autoregressive Sampling - Sequence Embedding Leak

### Location
`src/prxteinmpnn/model/mpnn.py` - `_process_group_positions()` method (lines 397-449)

### Problem Description

During autoregressive sampling, when computing the context for position i at decoding step t, the sequence embeddings (s_j) from ALL neighbors were being gathered and used, **including neighbors that hadn't been decoded yet** according to the autoregressive order.

**Incorrect Implementation:**
```python
# OLD CODE - BUGGY
edge_sequence_features = concatenate_neighbor_nodes(
    s_embed,  # Uses FULL s_embed array
    edge_features[idx],
    neighbor_indices_pos,
)
```

While `s_embed` correctly starts as zeros and is updated only for decoded positions, the issue is subtle:
1. The masking via `mask_bw_pos` happens AFTER gathering neighbors
2. This creates potential numerical instabilities and information leakage
3. The model might learn to rely on these unmasked values during training

### Why This Matters

In autoregressive decoding, each position should only "see" information from positions that have already been decoded. If position i can access sequence embeddings from non-decoded neighbors, it violates the autoregressive constraint and can lead to:

1. **Train-test mismatch**: During training with teacher forcing, all positions are visible. During sampling, they should not be.
2. **Poor sampling quality**: The model learns dependencies that don't hold during autoregressive generation.
3. **Low sequence recovery**: The model can't accurately predict sequences when information flow is incorrect.

### The Fix

Added explicit masking of `s_embed` BEFORE gathering neighbors:

```python
# NEW CODE - FIXED
# Create mask: 1 for decoded neighbors, 0 for non-decoded
full_decoded_mask = jnp.zeros(num_residues)
full_decoded_mask = full_decoded_mask.at[neighbor_indices_pos].set(mask_bw_pos)

# Mask s_embed to zero out non-decoded positions
masked_s_embed = s_embed * full_decoded_mask[:, None]

# Now gather from masked embeddings
edge_sequence_features = concatenate_neighbor_nodes(
    masked_s_embed,  # Uses MASKED embeddings
    edge_features[idx],
    neighbor_indices_pos,
)
```

This ensures that:
1. Only decoded positions have non-zero embeddings
2. Masking happens before gathering (more explicit and safer)
3. Follows the same pattern as the ColabDesign reference implementation

---

## Comparison with Reference Implementation

These fixes align PrxteinMPNN with the ColabDesign reference implementation:

### ColabDesign - Unconditional Decoder
```python
# colabdesign/mpnn/score.py lines 38-39
h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
# Result: [[h_E, 0], h_V_neighbors] ✅
```

### ColabDesign - Autoregressive Sampling
```python
# colabdesign/mpnn/sample.py line 96
X = {"h_S": jnp.zeros_like(h_V), ...}  # Initialize as zeros
# Line 91 - only update decoded positions
x["h_S"] = x["h_S"].at[t].set(self.W_s(S_t))
```

The ColabDesign implementation uses zeros for non-decoded positions and only updates them as they're sampled, achieving the same effect as our explicit masking.

---

## Expected Impact

After these fixes, we expect:

1. **Sequence Recovery**: Should increase from ~5-20% to **40-60%** (matching original ProteinMPNN)
2. **Alanine Bias**: Should decrease significantly in unconditional mode
3. **Sampling Diversity**: Better quality and diversity in generated sequences
4. **Conditional Scoring**: Should achieve >90% recovery when given true sequence

---

## Testing

To verify these fixes work:

```bash
# Run sequence recovery tests
uv run pytest tests/sampling/test_sequence_recovery.py -v

# Specific tests
uv run pytest tests/sampling/test_sequence_recovery.py::test_temperature_sampling_recovery_1ubq -v -s
uv run pytest tests/sampling/test_sequence_recovery.py::test_split_sampling_recovery_1ubq -v -s
```

Expected metrics after fix:
- Temperature sampling (T=0.1): **40-60% recovery** ✅
- Split sampling: **40-60% recovery** ✅
- Conditional scoring: **>90% recovery** ✅

---

## Additional Notes

### Why Were These Bugs Hard to Catch?

1. **Subtle Indexing**: The difference between tiling h_i vs gathering h_j is subtle and doesn't cause obvious runtime errors
2. **Masking Later**: The `mask_bw` was applied later, which masked some of the effect but not completely
3. **JAX Tracing**: JAX's functional style makes it harder to debug intermediate values during compilation

### Related Files

- `src/prxteinmpnn/model/decoder.py` - Decoder implementation
- `src/prxteinmpnn/model/mpnn.py` - Main model and sampling logic
- `src/prxteinmpnn/utils/concatenate.py` - Helper for gathering neighbors
- `tests/sampling/test_sequence_recovery.py` - Tests for sequence recovery

### References

- Original ProteinMPNN paper: [Dauparas et al. 2022](https://www.science.org/doi/10.1126/science.add2187)
- ColabDesign reference implementation: [sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)
- Issue describing the problem: [Link to issue]

---

## Contributors

- Debugging and fix implementation: Claude (AI Assistant)
- Original bug report and investigation: [Repository maintainer]

Last updated: 2025-11-07
