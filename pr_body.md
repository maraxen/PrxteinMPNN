## Summary

Fixed **3 critical bugs** causing extremely low sequence recovery (~5-20% instead of expected 40-60%). These bugs were identified by systematic comparison with the ColabDesign reference implementation and verification against the original working functional code.

## Bugs Fixed

### 🔴 Bug #1: Conditional Decoder Attention Mask (CRITICAL)
**Location**: `src/prxteinmpnn/model/decoder.py:376-381`

**Problem**: The conditional decoder was passing `attention_mask` to the decoder layer, which caused ALL messages (including encoder context) to be zeroed out for not-yet-decoded neighbors.

**Impact**: Conditional scoring ~30% recovery instead of >90%

**Fix**: Removed `attention_mask` parameter from layer call. The masking is already handled by `mask_bw` multiplication on line 369.

**Reference**: ColabDesign `score.py:76` does NOT pass attention mask to decoder layer.

---

### 🔴 Bug #2: Autoregressive Mask Uses Wrong Inequality (IMPORTANT)
**Location**: `src/prxteinmpnn/utils/autoregression.py:213`

**Problem**: Used `>=` allowing positions to attend to themselves during autoregressive decoding.

**Impact**: Positions could see their own sequence embeddings, violating autoregressive causality.

**Fix**: Changed from `>=` to `>` (strict inequality), excluding diagonal.

**Reference**: ColabDesign uses `jnp.tri(L, k=-1)` where `k=-1` explicitly excludes diagonal.

---

### 🔴 Bug #3: Autoregressive Encoder Context (MOST CRITICAL)
**Location**: `src/prxteinmpnn/model/mpnn.py:705-721`

**Problem**: Encoder context was using **central node features** (`h_i`) instead of **neighbor features** (`h_j`):

```python
# BEFORE (BUGGY)
encoder_context = jnp.concatenate([
  jnp.tile(jnp.expand_dims(node_features, -2), ...),  # ❌ h_i tiled
  encoder_edge_neighbors,
], -1)
```

This meant all neighbors provided the same structural information (central node's features), defeating the purpose of graph message passing.

**Impact**: Temperature sampling ~5% recovery (essentially random)

**Fix**: Changed to gather neighbor encoder features:

```python
# AFTER (FIXED)
encoder_context = concatenate_neighbor_nodes(
  node_features,  # ✓ Gathers h_j = node_features[neighbors]
  encoder_edge_neighbors,
  neighbor_indices,
)
```

**Root Cause**: Bug introduced during Equinox migration. The original functional code had this correct.

**Reference**:
- ColabDesign `sample.py:51` uses `cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)` which gathers neighbor features
- Original PrxteinMPNN functional code (before Equinox) had the same correct implementation

---

## Expected Impact

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Temperature Sampling (T=0.1) | ~5% | 40-60% ✅ |
| Conditional Scoring | ~30% | >90% ✅ |
| Split Sampling | ~20% | 40-60% ✅ |

## Files Changed

- `src/prxteinmpnn/model/decoder.py` - Fixed conditional decoder
- `src/prxteinmpnn/utils/autoregression.py` - Fixed AR mask generation
- `src/prxteinmpnn/model/mpnn.py` - Fixed autoregressive encoder context
- `SEQUENCE_RECOVERY_BUG_FIXES.md` - Comprehensive documentation
- `AUTOREGRESSIVE_ENCODER_CONTEXT_BUG_FIX.md` - Detailed encoder context fix docs
- `test_ar_mask_fix.py` - Verification test for AR mask

## Testing

```bash
# Verify AR mask fix
uv run python test_ar_mask_fix.py

# Run sequence recovery tests
uv run pytest tests/sampling/test_sequence_recovery.py -v
```

## Verification

✅ Compared line-by-line with ColabDesign reference implementation
✅ Verified against original working functional PrxteinMPNN code
✅ AR mask fix confirmed to exclude diagonal correctly
✅ All three bugs are now fixed and documented

## References

- ColabDesign: https://github.com/sokrypton/ColabDesign
- Key files compared:
  - `colabdesign/mpnn/score.py` (conditional decoding)
  - `colabdesign/mpnn/sample.py` (autoregressive sampling)
  - `colabdesign/mpnn/utils.py` (AR mask generation)
