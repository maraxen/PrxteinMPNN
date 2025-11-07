# Decoder Context Structure Comparison

This document provides a visual comparison of the decoder context structure before and after the bug fix.

## Visual Representation

### Unconditional Decoder Context

#### BEFORE (Buggy) ❌
```
Position i with K neighbors [n1, n2, n3]

For each neighbor k:
  context[i, k, :] = [h_i, 0, e_ik]
                      ^^^
                      Central node repeated!

Result: Every neighbor position sees the SAME central node features
```

**Visual:**
```
        h_i   0   e_i1
        h_i   0   e_i2     ← All use h_i (central)
        h_i   0   e_i3
        h_i   0   e_i4
```

#### AFTER (Fixed) ✅
```
Position i with K neighbors [n1, n2, n3]

For each neighbor k:
  context[i, k, :] = [e_ik, 0, h_k]
                                ^^^
                                Neighbor node features!

Result: Each neighbor position contributes its own features
```

**Visual:**
```
        e_i1  0   h_1
        e_i2  0   h_2     ← Each uses its own h_k
        e_i3  0   h_3
        e_i4  0   h_4
```

---

## Context Dimension Breakdown

### Feature Dimensions
- `h_i, h_j`: Node features (128-dim) from encoder
- `e_ij`: Edge features (128-dim) from encoder
- `0`: Zero padding (128-dim)
- **Total context**: 384-dim per neighbor

### Structure Comparison

| Component | Buggy | Fixed | Meaning |
|-----------|-------|-------|---------|
| **1st 128-dim** | h_i | e_ij | Edge geometry features |
| **2nd 128-dim** | 0 | 0 | Reserved for sequence (unconditional) |
| **3rd 128-dim** | e_ij | h_j | Neighbor node features |

---

## Conditional Decoder Context

For conditional decoding (scoring with known sequence), the structure is:

```
context[i, k, :] = [e_ik, s_k, h_k]
                         ^^^
                         Sequence embedding of neighbor k
```

Where:
- `s_k` = sequence embedding for neighbor k
- Only visible if neighbor k has been "decoded" according to AR mask
- Masked to 0 if neighbor k hasn't been decoded yet

---

## Information Flow

### Buggy Implementation
```
Position i sees:
  - Its own features (h_i) repeated K times
  - Edge geometries to neighbors (e_ik)
  - NO information about neighbor residue types or features

This is like asking "what should be here?" while blindfolded to your neighbors!
```

### Fixed Implementation
```
Position i sees:
  - Edge geometries to neighbors (e_ik)
  - Neighbor node features (h_k)
  - Neighbor sequence embeddings (s_k) if decoded

This is the correct message-passing: aggregate information FROM neighbors!
```

---

## Code Comparison

### Buggy Code
```python
# Tile central node for all neighbors
nodes_expanded = jnp.tile(
    jnp.expand_dims(node_features, -2),    # h_i -> (N, 1, 128)
    [1, edge_features.shape[1], 1],        # -> (N, K, 128)
)

# Concatenate: [h_i, 0, e_ij]
layer_edge_features = jnp.concatenate(
    [nodes_expanded, zeros_expanded, edge_features], -1
)
```

**Problem**: `nodes_expanded[i, k, :] = h_i` for ALL k (same central node)

### Fixed Code
```python
# Gather neighbor nodes
temp_context = concatenate_neighbor_nodes(
    jnp.zeros_like(node_features),  # 0
    edge_features,                   # e_ij
    neighbor_indices,                # where to gather
)  # -> [e_ij, 0]

# Gather neighbor features
layer_edge_features = concatenate_neighbor_nodes(
    node_features,      # h (will gather h[neighbor_indices])
    temp_context,       # [e_ij, 0]
    neighbor_indices,
)  # -> [[e_ij, 0], h_j]
```

**Solution**: `concatenate_neighbor_nodes` properly gathers `node_features[neighbor_indices[i, k]] = h_k`

---

## Impact on Model Behavior

### Without Neighbor Features (Buggy)
- **Cannot** learn from local structural context
- **Cannot** distinguish different neighbor environments
- **Must** predict based solely on position's own encoder output
- Results in **low sequence recovery** and **high Alanine bias**

### With Neighbor Features (Fixed)
- **Can** aggregate information from surrounding residues
- **Can** learn which amino acids fit the local environment
- **Can** implement proper message-passing as designed
- Results in **correct sequence recovery** (40-60%)

---

## Why This Bug Was Critical

1. **Breaks Message Passing**: The core idea of MPNN is to pass messages between nodes. Using h_i instead of h_j means NO messages are passed!

2. **Violates Architecture**: ProteinMPNN is designed as a message-passing neural network. This bug fundamentally broke that design.

3. **Explains Low Recovery**: Without neighbor information, the model cannot learn the sequence-structure relationships that make ProteinMPNN work.

---

## Verification

To verify the fix is correct, check that:

1. ✅ `layer_edge_features` has shape `(N, K, 384)`
2. ✅ For each position i, each of K neighbors contributes different features
3. ✅ `layer_edge_features[i, k, 256:384]` equals `node_features[neighbor_indices[i, k]]`
4. ✅ Sequence recovery improves to 40-60% on test cases

---

## Related Documentation

- [BUG_FIXES_SEQUENCE_RECOVERY.md](./BUG_FIXES_SEQUENCE_RECOVERY.md) - Full bug report and fixes
- [ColabDesign Reference](https://github.com/sokrypton/ColabDesign) - Reference implementation
- [ProteinMPNN Paper](https://www.science.org/doi/10.1126/science.add2187) - Original architecture

Last updated: 2025-11-07
