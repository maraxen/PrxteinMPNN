# Fix: Resolve Conditional and Autoregressive Decoder Bugs + Add Equivalence Tests

## Summary

This PR fixes critical bugs in the conditional and autoregressive decoding paths and adds comprehensive equivalence tests against ColabDesign. All three decoding paths now achieve **>0.95 correlation** with the original ColabDesign implementation! 🎉

## Changes

### 🐛 Bug Fixes

#### 1. Double Masking in Conditional Decoder (0.872 → 0.984 correlation)
**Problem**: The conditional decoder was applying attention masking twice:
1. When constructing `layer_edge_features` (correct)
2. Inside decoder layers by multiplying messages by `attention_mask` (incorrect)

When `ar_mask=0`, this zeroed out all messages, preventing decoder updates and causing the conditional path to diverge from unconditional.

**Solution**: Removed the `attention_mask` parameter from decoder layer calls in conditional path (`decoder.py:354-359`). The masking is already encoded in the edge features passed to each layer.

**Files Modified**: `src/prxteinmpnn/model/decoder.py`

#### 2. Wrong Encoder Context in Autoregressive Path (0.218 → 0.970 correlation)
**Problem**: Autoregressive sampling was constructing encoder context as `[h_i, e_ij, 0_j]` instead of `[e_ij, 0_j, h_j]`. This caused the first AR step (position 0 with no context) to produce completely different logits from unconditional (diff 5.25).

**Solution**: Changed encoder context construction in `mpnn.py:708-720` to properly use `concatenate_neighbor_nodes`, matching the unconditional decoder structure.

**Files Modified**: `src/prxteinmpnn/model/mpnn.py`

### ✅ Test Suite

Added comprehensive equivalence tests in `tests/model/test_colabdesign_equivalence.py`:

| Test | Description | Result |
|------|-------------|--------|
| `test_unconditional_logits` | Structure-based predictions | ✅ 0.984 correlation |
| `test_conditional_logits` | Fixed sequence scoring | ✅ 0.958-0.984 correlation |
| `test_autoregressive_sampling` | Sequential generation | ✅ 0.953-0.970 correlation |
| `test_ar_first_step_matches_unconditional` | AR sanity check | ✅ Pass |
| `test_conditional_with_zero_mask_matches_unconditional` | Conditional sanity check | ✅ Pass |

**All tests pass**: `5 passed in 72.56s`

### 📚 Documentation

- Updated `FINAL_VALIDATION_RESULTS.md` with comprehensive validation results
- Added CI test instructions
- Documented ColabDesign installation requirements
- Moved debug scripts to `debug_scripts/` directory for organization

### 🧹 Cleanup

Removed 60+ temporary debugging and test files from the codebase:
- Temporary test scripts used during debugging
- Duplicate debug scripts
- Intermediate validation markdown files
- Kept only essential documentation and three key debug scripts

### 🔧 Test Infrastructure

- Updated `pyproject.toml` with ColabDesign installation instructions
- Fixed `tests/model/test_mpnn.py` to pass `tie_group_map` parameter
- Renamed `tests/utils/test_concatenate.py` to avoid pytest collection conflicts

## Test Results

### Equivalence Tests
```bash
$ pytest tests/model/test_colabdesign_equivalence.py -v
========================= 5 passed in 72.56s =========================
```

### Model & Sampling Tests
```bash
$ pytest tests/model tests/sampling -q
========================= 31 passed, 2 skipped in 153.15s =========================
```

## Technical Details

### Root Cause Analysis

1. **Conditional Bug**: The conditional decoder was designed to support both full-context (ar_mask=1) and no-context (ar_mask=0) modes. However, the double masking bug caused all decoder messages to be zeroed when ar_mask=0, effectively preventing the decoder from updating node features. This is why conditional with ar_mask=0 diverged from unconditional.

2. **Autoregressive Bug**: The encoder context in the autoregressive path was incorrectly structured. The proper structure should match the unconditional decoder: `[edge_features, zeros, neighbor_node_features]`. Instead, it was `[tiled_node_features, edge_features, zeros]`, which caused completely different outputs.

### Validation Approach

All three decoding paths were systematically validated:
1. **Unconditional**: Pure structure-based, no sequence input
2. **Conditional**: Fixed sequence with ar_mask controlling attention
3. **Autoregressive**: Sequential sampling with Gumbel-max

The target correlation of >0.95 with ColabDesign was chosen to ensure near-identical behavior while accounting for minor numerical differences between JAX/Equinox (PrxteinMPNN) and dm-haiku (ColabDesign).

## Commits

1. `b3913b7` - refactor(constants): Replace hardcoded indices with proper atom ordering scheme
2. `b6a3031` - test: Add comprehensive validation tests for all decoding paths
3. `bdaa346` - **fix: Resolve conditional and autoregressive decoder bugs** (main fix)
4. `9353438` - debug: Add debugging scripts used to identify decoder bugs
5. `a40e806` - feat: Add ColabDesign equivalence tests and cleanup codebase

## Breaking Changes

None. The API remains unchanged; only internal implementation bugs were fixed.

## Dependencies

For running equivalence tests, ColabDesign must be installed:
```bash
pip install git+https://github.com/sokrypton/ColabDesign.git@e31a56f
```

This is documented in `pyproject.toml` under test dependencies.

## References

- Original ColabDesign implementation: https://github.com/sokrypton/ColabDesign
- Debug scripts showing bug discovery: `debug_scripts/`
- Full validation results: `FINAL_VALIDATION_RESULTS.md`
