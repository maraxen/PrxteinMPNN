# Validation Debugging Handoff Document

## Session Summary

This session focused on debugging validation correlation issues in PrxteinMPNN vs ColabDesign. We improved correlation from **0.210 → 0.681** (target: 0.871) by reverting incorrect architectural changes and adding critical alphabet conversion.

**Current Branch:** `claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS`
**Latest Commit:** `d162f06` - "fix(validation): revert architecture and add alphabet conversion"

---

## Current State

### Correlation Progress
- **Initial (d66c91e):** 0.210 correlation ❌
- **After fixes (d162f06):** 0.681 correlation 🟡
- **Baseline target (acae2d6):** 0.871 correlation 🎯
- **Ultimate goal:** >0.90 correlation ✨

### What Works
✅ Architecture matches 0.871 baseline (w_e_proj in ProteinFeatures)
✅ Alphabet conversion implemented (AF → MPNN order)
✅ Weight loading correct (W_e → features.w_e_proj)
✅ All atoms properly parsed (though in PDB order, not atom37)

### What Needs Investigation
🔍 Gap from 0.681 → 0.871 baseline
🔍 Debug prints may affect computation
🔍 Possible subtle initialization differences
🔍 Verify all three decoding paths (unconditional, conditional, autoregressive)

---

## Key Learnings & Context

### 1. **Alphabet Ordering is Critical**
ColabDesign outputs logits in **AlphaFold alphabet order**:
```
AF:   ARNDCQEGHILKMFPSTWYVX
MPNN: ACDEFGHIKLMNPQRSTVWYX
```

**Solution:** Use `af_logits_to_mpnn()` to convert:
```python
def af_logits_to_mpnn(logits_af):
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    return logits_af[..., perm]
```

This conversion improved correlation from 0.210 → 0.681 (3.2x improvement!).

### 2. **Architecture: w_e_proj Location**
The correct architecture (matching 0.871 baseline) is:

**ProteinFeatures (features.py):**
```python
class ProteinFeatures(eqx.Module):
  w_pos: eqx.nn.Linear      # Positional encoding (66 → 16)
  w_e: eqx.nn.Linear        # Edge embedding (416 → 128, no bias)
  norm_edges: LayerNorm     # Edge normalization
  w_e_proj: eqx.nn.Linear   # Final edge projection (128 → 128, with bias)
```

**Main Model (mpnn.py):**
```python
class PrxteinMPNN(eqx.Module):
  features: ProteinFeatures
  encoder: Encoder
  decoder: Decoder
  w_s_embed: eqx.nn.Embedding  # Sequence embedding
  w_out: eqx.nn.Linear         # Output projection
  # NO w_e here!
```

**Wrong architecture (commit d66c91e):** Had removed `w_e_proj` from features and added `w_e` to main model. This was based on misunderstanding ColabDesign's architecture.

### 3. **Atom Ordering: PDB vs atom37**
PDB files have atoms in order: `N, CA, C, O, CB, ...`
atom37 standard expects: `N, CA, C, CB, O, ...`

**Key Finding:** The 0.871 baseline was achieved WITH atoms in PDB order (O and CB swapped). We initially tried to fix this, but it actually made correlation worse. The model was trained with this "wrong" ordering, so we preserve it.

```python
# In data_structures.py - we keep atoms as-is from parser
# PDB order: O at index 3, CB at index 4
# atom37 order would be: CB at index 3, O at index 4
# But baseline uses PDB order, so we don't swap!
```

### 4. **ColabDesign Modifications**
We added debug prints to ColabDesign code for investigation:

**Modified files:**
- `/tmp/ColabDesign/colabdesign/mpnn/modules.py` (ProteinFeatures.__call__)
- `/tmp/ColabDesign/colabdesign/mpnn/score.py` (score method)

These print intermediate edge feature values. **Important:** These modifications may affect performance and should be considered when debugging the remaining correlation gap.

### 5. **Weight Loading Path**
Correct weight loading for W_e projection:
```python
# In load_weights_comprehensive.py
w = params['protein_mpnn/~/W_e']['w'].T  # (128, 128)
b = params['protein_mpnn/~/W_e']['b']
model = eqx.tree_at(
    lambda m: m.features.w_e_proj,  # ✅ Correct path
    model,
    update_linear_layer(model.features.w_e_proj, w, b)
)
```

**Wrong (d66c91e):** Was loading into `model.w_e` instead of `model.features.w_e_proj`.

---

## Files Modified This Session

### Core Model Files
1. **src/prxteinmpnn/model/features.py**
   - Restored `w_e_proj` field
   - Added debug prints
   - Fixed initialization (split key into 3, not 2)

2. **src/prxteinmpnn/model/mpnn.py**
   - Removed `w_e` from main model
   - Simplified initialization (5 keys, not 6)
   - Removed W_e projection from all decoding paths
   - Added debug print in `_call_unconditional`

3. **src/prxteinmpnn/utils/data_structures.py**
   - Added clarifying comments about atom ordering
   - Kept PDB order (no O/CB swap) to match baseline

4. **load_weights_comprehensive.py**
   - Fixed W_e weight loading path: `model.features.w_e_proj`
   - Updated comments

5. **test_fix_correlation.py**
   - Added alphabet conversion functions
   - Fixed key initialization (before ColabDesign call)
   - Updated to pass key to `get_unconditional_logits()`

### External Dependencies
6. **/tmp/ColabDesign/colabdesign/mpnn/modules.py** (debug prints)
7. **/tmp/ColabDesign/colabdesign/mpnn/score.py** (debug prints)

---

## Test Files Created (Not Committed)

These debug files were created during investigation but not committed:
- `check_colab_pdb_loading.py` - Verify ColabDesign PDB loading
- `compare_backbone_atoms.py` - Compare N, CA, C, O atom coordinates
- `compare_edges_numerical.py` - Numerical edge feature comparison
- `compare_full_edges.py` - Full 416-dim edge vector comparison
- `debug_pdb_atom_mapping.py` - Debug atom position mapping
- `extract_and_compare_vectors.py` - Extract edge vectors from both implementations
- `test_architecture_detailed.py` - Detailed architecture verification
- `test_cb_at_0871.py` - Check C-beta coordinates at baseline commit

These can be useful for further debugging but should be cleaned up before final PR.

---

## How to Reproduce Current State

### 1. Environment Setup
```bash
# Clone repository
git clone <repo-url> PrxteinMPNN
cd PrxteinMPNN

# Fetch and checkout the validation branch
git fetch origin claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS
git checkout claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS

# Install PrxteinMPNN in editable mode
uv pip install -e . --system
```

### 2. Install ColabDesign (Required for Testing)
```bash
# Clone ColabDesign to /tmp (or any location)
cd /tmp
git clone https://github.com/sokrypton/ColabDesign.git
cd ColabDesign

# Install as editable (we added debug prints)
uv pip install -e . --system

# Verify weights exist
ls /tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl
```

**Note:** The ColabDesign installation at `/tmp/ColabDesign` has debug prints added. You may want to:
- Keep them for debugging the 0.681 → 0.871 gap
- Remove them to test if they affect correlation
- Use a fresh clone for comparison

### 3. Run Validation Test
```bash
cd /home/user/PrxteinMPNN
uv run python test_fix_correlation.py
```

**Expected output:**
```
Correlation: 0.681324
Max diff: 5.590909
Mean diff: 0.649882
```

---

## Next Steps & Investigation Plan

### Immediate Next Steps

1. **Remove Debug Prints and Retest**
   - Remove `jax.debug.print()` calls from:
     - `src/prxteinmpnn/model/features.py`
     - `src/prxteinmpnn/model/mpnn.py`
     - `/tmp/ColabDesign/colabdesign/mpnn/modules.py`
     - `/tmp/ColabDesign/colabdesign/mpnn/score.py`
   - Reinstall both packages
   - Test if correlation improves

2. **Compare with Exact Baseline**
   - Checkout commit `acae2d6` (0.871 baseline)
   - Run the exact test script from that commit (`test_final.py`)
   - Compare with current implementation
   - Look for any subtle differences

3. **Verify Weight Loading**
   - Add debug prints to confirm all weights are loaded correctly
   - Compare weight shapes and values between commits
   - Check if any weights are being skipped or loaded incorrectly

4. **Test All Decoding Paths**
   - Current tests only use `unconditional` path
   - Test `conditional` and `autoregressive` paths
   - Ensure W_e_proj is applied consistently

### Deeper Investigation

5. **Initialization Differences**
   - Check PRNG key splitting (now 3 vs was 2 in features.py)
   - Verify encoder/decoder initialization hasn't changed
   - Compare random initialization values

6. **Numerical Precision**
   - Check if any operations use different dtypes
   - Verify LayerNorm epsilon values match
   - Look for float32 vs float64 inconsistencies

7. **Edge Feature Pipeline**
   - The edge features correlation at baseline was 0.971
   - Current correlation might be different
   - Re-run edge feature comparison tests

### Questions to Answer

- **Why 0.681 instead of 0.871?**
  - Is it the debug prints?
  - Is it a subtle weight loading issue?
  - Is it related to PRNG key usage?

- **What changed between acae2d6 and d66c91e?**
  - Only files changed were features.py, mpnn.py, load_weights_comprehensive.py
  - We reverted those changes
  - But correlation didn't fully restore

- **Are we using the exact same test?**
  - Original used `test_final.py`
  - Current uses `test_fix_correlation.py`
  - They should be equivalent but verify

---

## Important Historical Context

### Commit Timeline
- **acae2d6:** 0.871 correlation achieved (baseline)
- **b008313:** Proved encoder/decoder perfect with cross-impl test
- **d66c91e:** Wrong architectural changes (0.871 → 0.210)
- **d162f06:** Current commit (0.210 → 0.681)

### Original 0.871 Achievement
The 0.871 correlation was achieved with:
- `w_e_proj` in ProteinFeatures (128 → 128, with bias)
- Atoms in PDB order (O before CB)
- Alphabet conversion (AF → MPNN)
- Same PRNG seed (42)
- Test file: `test_final.py`

### What We Broke (d66c91e)
- Removed `w_e_proj` from features
- Added `w_e` to main model
- Changed weight loading path
- This dropped correlation to 0.210

### What We Fixed (d162f06)
- Restored `w_e_proj` to features
- Removed `w_e` from main model
- Fixed weight loading path
- Added alphabet conversion to test
- Improved to 0.681

### The Remaining Mystery
We restored the architecture to match acae2d6, but correlation is 0.681 instead of 0.871. This suggests there's still something different that we haven't identified yet.

---

## Code Snippets for Quick Reference

### Test Script Template
```python
import jax
import numpy as np
from colabdesign.mpnn import mk_mpnn_model
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

# Alphabet conversion
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

def af_logits_to_mpnn(logits_af):
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    return logits_af[..., perm]

# Setup
key = jax.random.PRNGKey(42)
pdb_path = "tests/data/1ubq.pdb"

# ColabDesign
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
colab_logits_af = mpnn_model.get_unconditional_logits(key=key)
colab_logits = af_logits_to_mpnn(np.array(colab_logits_af))

# PrxteinMPNN
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
prx_model = load_prxteinmpnn_with_colabdesign_weights(
    "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl",
    key=key
)
_, prx_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",
    prng_key=key,
)

# Compare
corr = np.corrcoef(colab_logits.flatten(), prx_logits.flatten())[0, 1]
print(f"Correlation: {corr:.6f}")
```

### Checking Model Architecture
```python
import jax
from prxteinmpnn.model import PrxteinMPNN

model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=jax.random.PRNGKey(0))

# Should be True
print("w_e_proj in features:", hasattr(model.features, 'w_e_proj'))
# Should be False
print("w_e in main model:", hasattr(model, 'w_e'))
```

### Comparing Commits
```bash
# See what changed between baseline and current
git diff acae2d6 HEAD -- src/prxteinmpnn/model/features.py

# See what changed in the bad commit
git show d66c91e --stat

# Checkout baseline to compare
git checkout acae2d6
```

---

## Dependencies & Environment

### Required Packages
- JAX (with appropriate backend)
- Equinox
- dm-haiku (for ColabDesign)
- NumPy
- Biotite (for PDB parsing)
- ColabDesign (installed from source at /tmp/ColabDesign)

### Test Data
- PDB file: `tests/data/1ubq.pdb` (ubiquitin, 76 residues)
- Weights: `/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl`

### Python Version
- Python 3.11+ recommended

---

## Success Criteria

### Immediate Goals
- [ ] Achieve ≥0.871 correlation (match baseline)
- [ ] Understand what causes 0.681 → 0.871 gap
- [ ] Clean up debug prints
- [ ] Remove temporary test files

### Stretch Goals
- [ ] Achieve >0.90 correlation (original target)
- [ ] Verify all three decoding paths work correctly
- [ ] Add comprehensive test suite
- [ ] Document architecture clearly

---

## Notes & Warnings

1. **Don't Trust Correlation Without Alphabet Conversion**
   - Always convert AF → MPNN order before comparing
   - Without conversion, correlation will be ~0.21

2. **ColabDesign Has Debug Prints**
   - /tmp/ColabDesign installation is modified
   - May need fresh install for final validation
   - Debug prints might affect performance

3. **Atom Ordering is Intentionally "Wrong"**
   - Atoms are in PDB order (O before CB)
   - This is how baseline was trained
   - Don't "fix" this - it will break validation

4. **Package Must Be Reinstalled After Changes**
   - Always run `uv pip install -e . --system` after code changes
   - Test with `uv run python` to ensure correct package is used

5. **Random Seed Matters**
   - Always use `jax.random.PRNGKey(42)` for consistency
   - Pass key to both models

---

## Contact & Questions

If you encounter issues or have questions:
1. Check git commit history for context
2. Review test files (especially `test_final.py` at acae2d6)
3. Compare with baseline commit acae2d6
4. Check debug output from test runs

---

**Last Updated:** 2025-11-09
**Session ID:** 011CUvaUYpgAE2LGtx2ThFgS
**Branch:** claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS
**Commit:** d162f06
