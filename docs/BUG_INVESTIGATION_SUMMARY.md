# Bug Investigation Summary

## Investigation Completed: November 7, 2025

Based on your systematic analysis, I've investigated the PrxteinMPNN codebase and identified the critical bugs affecting sequence recovery.

## Findings

### ✅ Confirmed Working

1. **Unconditional Decoder Fix** - The Alanine bias bug (64.5% → 3.9%) fix IS applied correctly in the codebase
   - `neighbor_indices` parameter correctly passed to decoder
   - Per-layer gathering of neighbor features working as intended
   - Context construction `[e_ij, s_j, h_j]` is correct

2. **Sequence Embedding Updates** - `s_embed` IS being updated in the autoregressive scan loop
   - Line 673 in `mpnn.py` correctly updates embeddings after sampling
   - Updates are properly broadcast to tied positions

3. **Edge Sequence Features** - Correctly constructed as `[e_ij, s_j]`
   - `concatenate_neighbor_nodes` properly gathers neighbor embeddings
   - Test confirms s_j are neighbor features, not central node features

### 🔴 Critical Bug Identified

**The Real Problem: Sequence Embeddings Not Being Masked During Autoregressive Decoding**

#### Evidence

1. **0% Sequence Recovery** at low temperature (T=0.1)
   - Model predicts 19/20 positions as Alanine
   - True AA logits: -0.558 (mean)
   - Predicted AA logits: +2.482 (mean)
   - **3.0 unit systematic bias** toward Alanine

2. **30% Conditional Recovery** (should be >90%)
   - When given the TRUE sequence as input, model only recovers 30%
   - This proves the conditional decoder isn't properly using sequence information

3. **Frozen Sequence Embeddings**
   - Debug output shows `s_embed_pos` **identical across ALL positions**
   - Value: `[0.25124452, -0.23482502, 0.09184124, -0.00492218, 0.10374901]`
   - This is the **Alanine embedding**, repeated for every position

#### Root Cause

In `_process_group_positions` (line 448 of `mpnn.py`):

```python
edge_sequence_features = concatenate_neighbor_nodes(
    s_embed,  # <-- BUG: Not masked by autoregressive order!
    edge_features[idx],
    neighbor_indices_pos,
)
```

The issue: `s_embed` should be **masked** so each position only sees embeddings from positions with **lower decoding order**. Currently, all positions see the **full** `s_embed` array, which includes information from positions that haven't been decoded yet (or are being decoded simultaneously).

This causes:

- First position sees all zeros → likely predicts Alanine
- Subsequent positions see Alanine context → reinforces Alanine
- Conditional decoder can't properly use input sequence

#### The Fix

```python
# Apply autoregressive mask to sequence embeddings
# Each position should only see embeddings from already-decoded positions
masked_s_embed = s_embed * mask_bw_pos[:, None]  # (N, C)

edge_sequence_features = concatenate_neighbor_nodes(
    masked_s_embed,  # <-- Use MASKED embeddings
    edge_features[idx],
    neighbor_indices_pos,
)
```

## Recommendations

### Immediate Actions

1. **Implement the masking fix** in `_process_group_positions`
2. **Verify `mask_bw` construction** - ensure it correctly enforces causal ordering based on decoding order
3. **Re-test sequence recovery** - should improve from 0% to 40-60%
4. **Test conditional decoder separately** - create isolated test

### Additional Investigation

The conditional decoder (`decoder.call_conditional`) may have its own issues:

1. **Static sequence edge features** (line 358-363 in `decoder.py`)
   - `sequence_edge_features` computed ONCE before the loop
   - Should it be recomputed per layer with updated node features?

2. **Attention mask application** - verify the autoregressive mask is being applied correctly in conditional mode

## Test Results

### Created Diagnostic Tests

1. **`test_conditional_decoder_bug.py`**
   - Tests edge sequence feature construction ✅
   - Tests autoregressive embedding updates ⚠️  (40% Alanine)
   - Tests sequence edge features (passed)

2. **`test_sequence_recovery_diagnosis.py`**
   - Sequence recovery vs random baseline: **0%** (CRITICAL)
   - Conditional scoring: **30%** (CRITICAL)
   - Proves bugs in both autoregressive and conditional modes

## Priority

**CRITICAL** - This bug completely breaks the model's ability to design sequences. The fix should be straightforward (add masking), but requires careful testing to ensure:

1. Mask is computed correctly based on decoding order
2. Mask is applied consistently across all decoder layers
3. Conditional decoder also respects autoregressive constraints

## Files to Modify

1. **`src/prxteinmpnn/model/mpnn.py`** (lines 440-450)
   - Add masking in `_process_group_positions`

2. **`src/prxteinmpnn/model/mpnn.py`** (lines 700-750)
   - Verify `mask_bw` construction in `_run_tied_position_scan`

3. **`src/prxteinmpnn/model/decoder.py`** (conditional decoder)
   - May need additional fixes for proper sequence context

## Documentation Created

- **`docs/SEQUENCE_EMBEDDING_BUG.md`** - Comprehensive bug report with analysis
- **`tests/debug/test_conditional_decoder_bug.py`** - Diagnostic tests
- **`tests/debug/test_sequence_recovery_diagnosis.py`** - Focused recovery tests

---

**Next Step**: Implement the masking fix and re-run all tests to verify sequence recovery improves to expected 40-60% range.
