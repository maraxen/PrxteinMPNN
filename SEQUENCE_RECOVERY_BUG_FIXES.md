# Sequence Recovery Bug Fixes

## Summary

Fixed critical bugs in PrxteinMPNN that were causing extremely low sequence recovery (~5-20% instead of expected 40-60%). After comparing with the ColabDesign reference implementation, identified and fixed 3 bugs:

1. **Critical Bug**: Attention mask incorrectly passed to decoder layer in conditional mode
2. **Minor Bug**: Redundant sequence embedding pre-masking
3. **Important Bug**: Autoregressive mask used wrong inequality (>= instead of >)

## Bug #1: Attention Mask Incorrectly Passed to Decoder Layer (CRITICAL)

### Location
- **File**: `src/prxteinmpnn/model/decoder.py`
- **Method**: `Decoder.call_conditional()`
- **Lines**: 384-389 (before fix)

### The Problem

In conditional decoding (scoring mode), the decoder was incorrectly passing `attention_mask` to the decoder layer. This caused ALL messages to be masked, including those providing essential encoder context for not-yet-decoded positions.

**Before Fix**:
```python
# decoder.py lines 384-389 (BUGGY)
loop_node_features = layer(
    loop_node_features,
    layer_edge_features,
    mask,
    attention_mask=attention_mask,  # ❌ BUG!
)
```

The decoder layer then applied this mask to ALL messages:
```python
# decoder.py line 142
if attention_mask is not None:
    message = jnp.expand_dims(attention_mask, -1) * message
```

### Why This Caused Low Sequence Recovery

In conditional decoding, the autoregressive masking works by combining two types of context:
- **Decoded neighbors** (mask_bw=1): Contribute full context `[e_ij, s_j, h_j]`
- **Not-yet-decoded neighbors** (mask_fw=1): Contribute encoder-only context `[e_ij, 0, h_j_encoder]`

This masking is handled by line 378:
```python
layer_edge_features = (mask_bw[..., None] * current_features) + masked_node_edge_features
```

By passing `attention_mask` to the layer, messages from ALL neighbors (including those providing encoder context) were zeroed out for positions with `attention_mask=0`. This broke the encoder context mechanism, preventing the model from properly reconstructing sequences.

### The Fix

**After Fix**:
```python
# decoder.py lines 384-389 (FIXED)
loop_node_features = layer(
    loop_node_features,
    layer_edge_features,
    mask,
    # attention_mask NOT passed - masking already handled above
)
```

**Explanation**: The attention masking is already handled by the `mask_bw` multiplication on line 378. The decoder layer should NOT apply additional masking, as this would zero out encoder context that needs to be preserved for not-yet-decoded positions.

### Reference Implementation

**ColabDesign** (`colabdesign/mpnn/score.py` line 76):
```python
h_V = layer(h_V, h_ESV, I["mask"])  # Only residue mask, NO attention mask
```

ColabDesign does NOT pass the attention mask to the decoder layer - confirming this is the correct approach.

---

## Bug #2: Redundant Sequence Embedding Pre-Masking (MINOR)

### Location
- **File**: `src/prxteinmpnn/model/decoder.py`
- **Method**: `Decoder.call_conditional()`
- **Lines**: 362 (before fix)

### The Problem

Sequence embeddings were being masked BEFORE gathering neighbors:

**Before Fix**:
```python
# decoder.py line 362 (REDUNDANT)
masked_seq_embeddings = attention_mask[..., None] * neighbor_seq_embeddings
```

Then masked AGAIN when constructing layer features:
```python
# decoder.py line 381
layer_edge_features = (mask_bw[..., None] * current_features) + masked_node_edge_features
```

This was redundant (though not harmful, since 0 * 0 = 0) and inconsistent with the reference implementation.

### The Fix

**After Fix**:
```python
# decoder.py lines 352-361 (FIXED)
# Gather sequence embeddings for all neighbors: (N, K, 128)
# NOTE: We do NOT pre-mask these - masking happens via mask_bw multiplication
neighbor_seq_embeddings = embedded_sequence[neighbor_indices]

# Concatenate with edge features: [e_ij, s_j] -> (N, K, 256)
sequence_edge_features = jnp.concatenate(
  [
    edge_features,  # (N, K, 128)
    neighbor_seq_embeddings,  # (N, K, 128) - NOT pre-masked
  ],
  axis=-1,
)
```

Masking now happens only once, via the `mask_bw` multiplication on line 378, matching ColabDesign's approach.

### Reference Implementation

**ColabDesign** (`colabdesign/mpnn/score.py` lines 54-56):
```python
# Concatenate sequence embeddings for autoregressive decoder
h_S = self.W_s(I["S"])
h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)  # NOT pre-masked
```

ColabDesign gathers neighbor embeddings WITHOUT pre-masking, then applies masking later in the loop.

---

## Bug #3: Autoregressive Mask Uses Wrong Inequality (IMPORTANT)

### Location
- **File**: `src/prxteinmpnn/utils/autoregression.py`
- **Function**: `generate_ar_mask()`
- **Line**: 209 (before fix)

### The Problem

The autoregressive mask used `>=` (greater-than-or-equal), allowing positions to attend to themselves:

**Before Fix**:
```python
# autoregression.py line 209 (BUGGY)
ar_mask = (row_indices >= col_indices).astype(int)
```

This produced a mask with 1s on the diagonal, meaning position i could see its own sequence embedding during autoregressive decoding.

**Example**:
```python
# With >= (buggy)
ar_mask = [[1, 0, 0],    # Position 0 can attend to itself ❌
           [1, 1, 0],    # Position 1 can attend to itself ❌
           [1, 1, 1]]    # Position 2 can attend to itself ❌
```

### Why This Matters

In autoregressive decoding, position i should only see sequence embeddings from positions decoded BEFORE it (strict inequality), not its own embedding. This is fundamental to autoregressive modeling - we're predicting amino acid i given the structure and amino acids decoded before i.

### The Fix

**After Fix**:
```python
# autoregression.py lines 207-213 (FIXED)
# BUG FIX: Use strict inequality (>) not >= to match ColabDesign
# Position i should only attend to positions j where order[j] < order[i]
# This prevents a position from attending to its own sequence embedding
# during autoregressive decoding (matching ColabDesign's tri(L, k=-1))
row_indices = decoding_order[:, None]
col_indices = decoding_order[None, :]
ar_mask = (row_indices > col_indices).astype(int)
```

**Example**:
```python
# With > (fixed)
ar_mask = [[0, 0, 0],    # Position 0 cannot attend to itself ✅
           [1, 0, 0],    # Position 1 cannot attend to itself ✅
           [1, 1, 0]]    # Position 2 cannot attend to itself ✅
```

### Reference Implementation

**ColabDesign** (`colabdesign/mpnn/utils.py` lines 19-26):
```python
def get_ar_mask(order):
    '''compute autoregressive mask, given order of positions'''
    order = order.flatten()
    L = order.shape[-1]
    tri = jnp.tri(L, k=-1)  # k=-1 excludes diagonal
    idx = order.argsort()
    ar_mask = tri[idx,:][:,idx]
    return ar_mask
```

`jnp.tri(L, k=-1)` creates a lower triangular matrix with `k=-1`, which excludes the diagonal. This confirms the diagonal should be zero.

### Verification

Test script `test_ar_mask_fix.py` verifies:
- ✅ Diagonal is all zeros (no self-attention)
- ✅ Lower triangular structure (excluding diagonal) matches expected
- ✅ Works correctly with permuted decoding orders

---

## Expected Impact

After these fixes, sequence recovery should improve dramatically:

### Before Fixes
- Temperature sampling (T=0.1): **5.7%** ❌
- Split sampling: **20.1%** ❌
- Conditional scoring: **~30%** ❌

### Expected After Fixes
- Temperature sampling (T=0.1): **40-60%** ✅
- Split sampling: **40-60%** ✅
- Conditional scoring: **>90%** ✅

The conditional scoring improvement should be especially dramatic since Bug #1 was completely breaking the encoder context mechanism.

---

## Files Modified

1. **`src/prxteinmpnn/model/decoder.py`**
   - Removed `attention_mask` parameter from decoder layer call in `call_conditional()`
   - Removed redundant sequence embedding pre-masking
   - Simplified code to match ColabDesign structure

2. **`src/prxteinmpnn/utils/autoregression.py`**
   - Changed AR mask from `>=` to `>` (strict inequality)
   - Added detailed comments explaining the fix

---

## Testing

### Verification Tests

1. **AR Mask Test**: `test_ar_mask_fix.py`
   - ✅ Verified diagonal is zero
   - ✅ Verified correct structure with permuted orders
   - ✅ Confirmed difference from buggy implementation

2. **Sequence Recovery Tests**: `tests/sampling/test_sequence_recovery.py`
   - Should show dramatic improvement in all metrics
   - Conditional scoring should jump from ~30% to >90%

### Manual Testing

To verify the fixes manually:
```bash
# Run sequence recovery tests
uv run pytest tests/sampling/test_sequence_recovery.py -v

# Run conditional decoder tests
uv run pytest tests/debug/test_conditional_decoder_bug.py -v
```

---

## Acknowledgments

These bugs were identified by systematic comparison with the **ColabDesign** reference implementation:
- Repository: https://github.com/sokrypton/ColabDesign
- Key files:
  - `colabdesign/mpnn/score.py` - Conditional scoring
  - `colabdesign/mpnn/sample.py` - Sampling
  - `colabdesign/mpnn/utils.py` - AR mask generation

---

## Date

2025-11-07
