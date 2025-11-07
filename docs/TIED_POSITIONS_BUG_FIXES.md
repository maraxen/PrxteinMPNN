# Tied Positions Bug Analysis and Fixes

## Summary

After thorough analysis of the PrxteinMPNN codebase, I identified **one critical bug** in the tied-position autoregressive sampling path. The initial analysis suggested two bugs, but upon investigation, only one was confirmed.

## Bug #1 (INITIAL REPORT): Temperature Strategy - **NOT A BUG**

### Initial Report
The initial analysis claimed that the "temperature" sampling strategy in `prxteinmpnn/sampling/sample.py` failed to implement tied-position logic because it just called the model once without a group-wise sampling loop.

### Investigation Result: **FALSE ALARM**

The model **DOES** have complete tied-position support built into its internal sampling logic:

1. `make_sample_sequences` with `sampling_strategy="temperature"` correctly passes `tie_group_map` and `num_groups` to the model
2. The model's `__call__` method dispatches to `_call_autoregressive`
3. `_call_autoregressive` passes `tie_group_map` to `_run_autoregressive_scan`
4. `_run_autoregressive_scan` checks `if tie_group_map is None` and routes to `_run_tied_position_scan` when groups are provided
5. `_run_tied_position_scan` implements the correct group-wise sampling with logit averaging (lines 450-600 in `src/prxteinmpnn/model/mpnn.py`)

**Conclusion:** The temperature strategy works correctly. No fix needed.

### Test Verification
```python
# tests/sampling/test_tied_positions_bugs.py::test_bug_temperature_strategy_tied_positions
# Status: PASSED ✓
```

---

## Bug #2 (CONFIRMED): Split-Sampling AR Mask - **CRITICAL BUG FIXED**

### Location
`prxteinmpnn/sampling/sample.py`, line 358 in `make_encoding_sampling_split_fn`

### The Bug
```python
# BEFORE (BUGGY):
autoregressive_mask = generate_ar_mask(decoding_order, tie_group_map)
```

The function `generate_ar_mask` has the signature:
```python
def generate_ar_mask(
  decoding_order: DecodingOrder,
  chain_idx: jnp.ndarray | None = None,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
) -> AutoRegressiveMask:
```

The buggy call passes `tie_group_map` as the **second argument** (`chain_idx`), not the third argument (`tie_group_map`). This causes:

1. **Incorrect Parameter Mapping:**
   - `tie_group_map` → `chain_idx` parameter (WRONG!)
   - `None` → `tie_group_map` parameter (WRONG!)
   - `None` → `num_groups` parameter (WRONG!)

2. **Broken AR Mask Logic:**
   - The code that handles tied positions (`if tie_group_map is None:`) is never triggered
   - Instead, the `if chain_idx is not None:` logic is triggered
   - This creates a mask where positions can only attend to previous positions in the same "chain" (actually the same tie group)
   - Positions cannot see any information from other groups, severely limiting the model's context

3. **Mask Comparison:**
   ```
   Correct AR mask (tied positions):
   [[1 1 0 0 0 0]    # Group 0: positions 0,1 can attend to each other
    [1 1 0 0 0 0]
    [1 1 1 1 0 0]    # Group 1: positions 2,3 can attend to group 0 AND each other
    [1 1 1 1 0 0]
    [1 1 1 1 1 1]    # Group 2: positions 4,5 can attend to all previous groups
    [1 1 1 1 1 1]]
   
   Buggy AR mask (tie_group_map as chain_idx):
   [[1 0 0 0 0 0]    # Positions can only attend to themselves
    [1 1 0 0 0 0]    # and earlier positions in the SAME group
    [0 0 1 0 0 0]    # Cannot see ANY other groups!
    [0 0 1 1 0 0]
    [0 0 0 0 1 0]
    [0 0 0 0 1 1]]
   ```

### The Fix
```python
# AFTER (FIXED):
autoregressive_mask = generate_ar_mask(decoding_order, None, tie_group_map, num_groups)
```

Now the parameters are correctly mapped:
- `decoding_order` → 1st parameter ✓
- `None` → `chain_idx` ✓
- `tie_group_map` → `tie_group_map` ✓
- `num_groups` → `num_groups` ✓

### Impact

**Before the fix:**
- The split-sampling path (`make_encoding_sampling_split_fn`) used a severely broken AR mask
- Positions in a tie group could only see themselves and earlier positions in the same group
- The model was starved of context from other groups
- Despite this, sequences were still tied (due to the group-wise sampling loop) but used logits computed with insufficient context

**After the fix:**
- The AR mask correctly allows positions to attend to all previous groups
- Positions within a group can attend to each other (same decoding step)
- The model has full context for computing logits

### Test Verification
```python
# tests/sampling/test_tied_positions_bugs.py::test_ar_mask_generation_with_chain_idx
# Status: PASSED ✓ (confirms bug and fix)

# tests/sampling/test_tied_positions_bugs.py::test_encoding_split_sample_fn_bug
# Status: PASSED ✓ (confirms sequences are tied after fix)
```

---

## Files Modified

1. **`src/prxteinmpnn/sampling/sample.py`**
   - Line 358: Fixed `generate_ar_mask` call in `make_encoding_sampling_split_fn`

2. **`tests/sampling/test_tied_positions_bugs.py`** (NEW)
   - Comprehensive test suite for tied-position bugs
   - Tests for both temperature strategy and split-sampling path
   - Demonstrates the AR mask bug with concrete examples

---

## Recommendation

The split-sampling path (`make_encoding_sampling_split_fn`) is currently marked as temporarily disabled in `run/sampling.py`:
```python
if spec.average_encodings:
    msg = "average_encodings feature is temporarily disabled during Equinox migration"
    raise NotImplementedError(msg)
```

**Since this code path was not actively used, the bug did not manifest in production.** However, with the fix applied, this feature can now be safely re-enabled.

---

## Additional Notes

### Why the bug was hard to detect:
1. The split-sampling function has its own group-wise sampling loop that enforces tied positions
2. Sequences were still correctly tied (same amino acids) despite the broken AR mask
3. The bug only affected the **quality of the logits** used for sampling, not whether positions were tied
4. The feature was disabled, so no production code was affected

### Code quality improvements implemented:
1. Added comprehensive test coverage for tied-position sampling
2. Tests now verify both that sequences are tied AND that the AR mask is correct
3. Tests demonstrate the difference between correct and buggy AR masks with concrete examples
