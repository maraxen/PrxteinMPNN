# Summary: Tied-Position Autoregressive Sampling Bug Fixes

## Overview

I investigated the tied-position autoregressive sampling bugs reported in PrxteinMPNN and found **one critical bug** that has been fixed. The initial report suggested two bugs, but detailed analysis revealed only one was real.

## Findings

### Bug #1: Temperature Strategy (REPORTED BUT NOT CONFIRMED) ❌

**Initial Report:** The "temperature" sampling strategy was claimed to lack tied-position support.

**Investigation Result:** **FALSE ALARM** - The model DOES have complete tied-position support:
- The model's `_run_autoregressive_scan` correctly routes to `_run_tied_position_scan` when `tie_group_map` is provided
- The `_run_tied_position_scan` method (lines 450-600 in `src/prxteinmpnn/model/mpnn.py`) implements proper group-wise sampling with logit averaging
- All tests pass, confirming correct behavior

**Status:** ✅ No fix needed - working as designed

---

### Bug #2: Split-Sampling AR Mask (CONFIRMED AND FIXED) ✅

**Location:** `src/prxteinmpnn/sampling/sample.py`, line 358

**The Bug:**
```python
# BEFORE (BUGGY):
autoregressive_mask = generate_ar_mask(decoding_order, tie_group_map)
```

The function signature is:
```python
def generate_ar_mask(
  decoding_order: DecodingOrder,
  chain_idx: jnp.ndarray | None = None,      # 2nd parameter
  tie_group_map: jnp.ndarray | None = None,  # 3rd parameter
  num_groups: int | None = None,             # 4th parameter
)
```

**Problem:** `tie_group_map` was passed as the 2nd argument (`chain_idx`) instead of the 3rd argument.

**Impact:**
- The tied-position logic (`if tie_group_map is None:`) was never triggered
- Instead, the chain-boundary logic was triggered with groups treated as chains
- This created an overly restrictive AR mask where positions could only attend to themselves and earlier positions **within the same group**
- Positions could not see information from other groups, starving the model of context

**Example:**
```
Correct AR mask:            Buggy AR mask:
[[1 1 0 0 0 0]              [[1 0 0 0 0 0]
 [1 1 0 0 0 0]               [1 1 0 0 0 0]
 [1 1 1 1 0 0]  <-- Can      [0 0 1 0 0 0]  <-- Cannot
 [1 1 1 1 0 0]      see       [0 0 1 1 0 0]      see
 [1 1 1 1 1 1]      all       [0 0 0 0 1 0]      other
 [1 1 1 1 1 1]]     groups    [0 0 0 0 1 1]]     groups!
```

**The Fix:**
```python
# AFTER (FIXED):
autoregressive_mask = generate_ar_mask(decoding_order, None, tie_group_map, num_groups)
```

**Status:** ✅ Fixed - all tests pass

---

## Files Modified

1. **`src/prxteinmpnn/sampling/sample.py`** (line 358)
   - Fixed incorrect function call to `generate_ar_mask`

2. **`tests/sampling/test_tied_positions_bugs.py`** (NEW)
   - Comprehensive test suite demonstrating the bug and verifying the fix
   - 4 tests covering temperature strategy, split-sampling, and AR mask generation

3. **`docs/TIED_POSITIONS_BUG_FIXES.md`** (NEW)
   - Detailed documentation of findings and fixes

---

## Test Results

All tests pass:
```
tests/sampling/test_tied_positions_bugs.py::test_bug_temperature_strategy_tied_positions PASSED
tests/sampling/test_tied_positions_bugs.py::test_bug_split_sampling_ar_mask PASSED
tests/sampling/test_tied_positions_bugs.py::test_encoding_split_sample_fn_bug PASSED
tests/sampling/test_tied_positions_bugs.py::test_ar_mask_generation_with_chain_idx PASSED
```

All existing sampling tests also pass (23 passed, 2 skipped).

---

## Why This Bug Was Hard to Detect

1. The split-sampling feature (`make_encoding_sampling_split_fn`) was disabled in production:
   ```python
   if spec.average_encodings:
       raise NotImplementedError("temporarily disabled during Equinox migration")
   ```

2. The `sample_fn` has its own group-wise sampling loop that enforces tied positions, so sequences were still correctly tied despite the broken AR mask

3. The bug only affected the **quality of the logits** used for sampling, not whether positions were tied

4. Without the fix, the model computed logits with insufficient context, but this is subtle and doesn't cause obvious errors

---

## Recommendation

With the bug fixed, the `average_encodings` feature can be safely re-enabled if needed.
