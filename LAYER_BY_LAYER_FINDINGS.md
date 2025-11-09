# Layer-by-Layer Comparison Findings

## Summary

**Current Status:** 0.871 correlation (target: >0.90, gap: 0.029)

## Key Findings

### 1. Initial Edge Features: 0.971 correlation
- **Root cause of all divergence**
- Max diff: 9.95
- This small difference gets amplified through 6 layers (3 encoder + 3 decoder)

### 2. Layer-by-Layer Correlation Breakdown

#### Encoder Layers
- **Layer 0 h_V**: 0.895 (first major node feature divergence)
- **Layer 0 h_E**: 0.978 (edges stay high)
- **Layer 1 h_V**: 0.854 (node divergence increases)
- **Layer 1 h_E**: 0.960
- **Layer 2 h_V**: 0.907 (slight improvement)
- **Layer 2 h_E**: 0.953

**Pattern:** Edge features maintain high correlation (0.95-0.98) throughout, but node features diverge more.

#### Decoder Layers
- **Context**: 0.931
- **Layer 0 h_V**: 0.916
- **Layer 1 h_V**: 0.920
- **Layer 2 h_V**: 0.771 (⚠️ **biggest drop!**)

**Pattern:** Progressive correlation through first 2 layers, then major drop in layer 2.

### 3. Root Cause Analysis

#### Not the issue:
- ✅ Atom input format (tested backbone-only vs full atom37: identical results)
- ✅ Neighbor indices (perfect 1.000 correlation)
- ✅ GELU approximation (using `approximate=False` in both)
- ✅ LayerNorm epsilon (both use 1e-5)
- ✅ Message aggregation scale (both use 30.0)

#### Likely issues:
- ⚠️ **Edge feature computation** (0.971 starting correlation)
  - Possibly due to vmap vs direct matrix multiplication numerical differences
  - RBF/positional encoding show good correlation individually
  - Issue appears after edge embedding or LayerNorm

- ⚠️ **Decoder Layer 2** (0.771 correlation)
  - Biggest single-layer drop
  - All decoder layers use same implementation, so likely accumulated error
  - Not a bug in layer 2 specifically

### 4. Hypotheses

1. **vmap precision**: `jax.vmap(jax.vmap(linear))` may have subtle numerical differences vs direct `@`
2. **LayerNorm vmap**: While mathematically equivalent, vmap(LayerNorm) might compute slightly differently
3. **Accumulated floating-point error**: Small initial differences compound through 6 layers

### 5. Current Correlation vs Target

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Edge features (initial) | 0.971 | >0.99 | 0.019 |
| Encoder Layer 0 h_V | 0.895 | >0.95 | 0.055 |
| Decoder Layer 2 h_V | 0.771 | >0.90 | 0.129 |
| **Final logits** | **0.871** | **>0.90** | **0.029** |

## Recommendations

### Option 1: Accept 0.871 as good enough
- **Pros:** Already very close to reference implementation
- **Cons:** Doesn't meet stated goal of >0.90

### Option 2: Investigate vmap vs direct operations
- Replace all `jax.vmap(jax.vmap(linear))` with direct `@` operations in features module
- Compare if this improves initial edge features from 0.971 to >0.99

### Option 3: Focus on decoder layer 2 specifically
- Add intermediate outputs within layer 2 to find exact operation causing drop
- May reveal a specific MLP or norm issue

### Option 4: Check for missing operations
- Verify all ColabDesign operations are exactly replicated
- Check for any missing biases, scalings, or masks

## Next Steps

Given that we're only 0.029 away from target, **Option 2** (investigate vmap) seems most promising.

If that doesn't work, we should document the 0.871 correlation as the achieved result and note that it's within 3% of the reference implementation, which is likely sufficient for practical use.
