# Critical Bug Report: Sequence Embedding Not Updating Properly

**Date:** November 7, 2025  
**Status:** 🔴 CRITICAL BUG IDENTIFIED  
**Impact:** Sequence recovery ~0-5% (should be 40-60%)

---

## Executive Summary

While the Alanine bias bug (64.5% → 3.9%) was successfully fixed in the unconditional decoder, a **CRITICAL BUG remains in the autoregressive sampling logic** that prevents proper sequence information from being used during sampling.

### Key Evidence

1. **Sequence recovery: 0%** at low temperature (T=0.1)
   - Model predicts **19/20 positions as Alanine**
   - Logits for true AAs: -0.558 (mean)
   - Logits for predicted AAs: +2.482 (mean)
   - **3.0 unit bias** toward Alanine

2. **Conditional decoder: Only 30% recovery** when given TRUE sequence
   - Should be >90% when model sees correct answer
   - Indicates conditional decoder not properly using sequence embeddings

3. **Sequence embeddings frozen at Alanine**
   - Debug output shows `s_embed_pos` is **identical across all positions**
   - Value: `[0.25124452, -0.23482502, 0.09184124, -0.00492218, 0.10374901]`
   - This is the Alanine (AA index 0) embedding vector

---

## Root Cause Analysis

### The Bug

The sequence embeddings `s_embed` **ARE** being updated in the carry state of the autoregressive scan loop (line 673 in `mpnn.py`):

```python
s_embed = jnp.where(group_mask[:, None], jnp.squeeze(s_embed_new), s_embed)
```

**HOWEVER**, there's a critical issue with **how the updated `s_embed` is being used**.

Looking at the debug output from the test, we can see that after each sampling step, the `s_embed_pos[:5]` values are printed, and they are **ALWAYS THE SAME** - they're always showing the Alanine embedding, regardless of what amino acid was just sampled.

### Why This Happens

The issue is in `_process_group_positions` (lines 405-490 in `mpnn.py`). Let me trace through what happens:

1. **Initial state**: `s_embed` is zeros (line 602)
2. **First group sampled**: Let's say we sample Alanine (AA=0)
   - `s_embed` gets updated to Alanine embedding for positions in that group
3. **Second group processed**: 
   - We pass the updated `s_embed` to `_process_group_positions`
   - Inside the `fori_loop` (line 438), we compute `edge_sequence_features`
   - **BUT** we're computing it using the current `s_embed` state

The problem is that during the `fori_loop` over positions within a group, we're using the **same** `s_embed` for all positions in that group. This `s_embed` was set based on the **previous** groups that were sampled, but within the current group, all positions see the same (potentially incorrect) sequence context.

### The Real Issue: Premature Sequence Context

In the autoregressive decoding, each position should only see the sequence embeddings of positions that have **already been decoded** (positions with lower decoding order). However, the current implementation passes the full `s_embed` array to each position, which includes:

1. **Zeros** for positions not yet decoded (correct)
2. **Sampled embeddings** for positions already decoded (correct)
3. **BUT** within a group being processed, all positions see the **SAME** sequence context

This means that when processing a group, the decoder can't distinguish between positions within that group because they all receive identical sequence information.

---

## The Fix

The fix involves ensuring that during autoregressive decoding, each position only sees sequence embeddings from positions with **lower decoding order**.

### Current Code (Buggy)

In `_process_group_positions`, line 448:

```python
edge_sequence_features = concatenate_neighbor_nodes(
    s_embed,  # <-- This is the FULL sequence embedding array
    edge_features[idx],
    neighbor_indices_pos,
)
```

The problem is that `s_embed` includes embeddings for ALL positions processed so far, not just the ones with lower decoding order than the current position.

### Proposed Fix

We need to mask `s_embed` based on the autoregressive mask before using it:

```python
# Mask sequence embeddings based on autoregressive order
# Only positions with lower decoding order should be visible
masked_s_embed = s_embed * mask_bw_pos[:, None]  # (N, C)

edge_sequence_features = concatenate_neighbor_nodes(
    masked_s_embed,  # <-- Use MASKED embeddings
    edge_features[idx],
    neighbor_indices_pos,
)
```

This ensures that each position only sees sequence information from positions that have already been decoded in the current autoregressive trajectory.

---

## Why This Explains All Symptoms

1. **Alanine bias** (even after decoder fix):
   - First position decoded is likely Alanine due to bias
   - All subsequent positions see Alanine context
   - This reinforces Alanine predictions throughout

2. **Low conditional recovery** (30%):
   - Conditional decoder should use ground truth sequence
   - But if masking is wrong, positions see incorrect context
   - Model can't properly condition on the input sequence

3. **Frozen sequence embeddings**:
   - Debug shows `s_embed_pos` not changing
   - This is because the autoregressive mask isn't being applied
   - All positions see the same (Alanine) context

---

## Implementation Plan

### Step 1: Fix `_process_group_positions`

Modify the autoregressive mask application to properly mask sequence embeddings before computing edge features.

**File:** `src/prxteinmpnn/model/mpnn.py`  
**Location:** Lines 440-450

### Step 2: Verify Autoregressive Mask Construction

Ensure `mask_bw` is correctly constructed to enforce causal ordering.

**File:** `src/prxteinmpnn/model/mpnn.py`  
**Location:** Lines 700-750 (in `_run_tied_position_scan`)

### Step 3: Test Conditional Decoder Separately

Create a test that directly calls `decoder.call_conditional` to verify it's working correctly in isolation.

---

##Next Steps

1. **Implement the mask fix** in `_process_group_positions`
2. **Re-run sequence recovery tests** to verify improvement
3. **Test conditional decoder** in isolation
4. **Compare with original ProteinMPNN implementation** for validation

---

## Expected Outcome

After fix:
- **Sequence recovery**: 40-60% (from 0-5%)
- **Conditional recovery**: >90% (from 30%)
- **Diversity**: High (not all Alanine)
- **Alanine bias**: <15% (from 100%)

---

## References

- DECODER_BUG_FIX_SUMMARY.md (previous Alanine bias fix)
- Original ProteinMPNN paper (Dauparas et al. 2022)
- ColabDesign implementation (for comparison)

**End of Report**
