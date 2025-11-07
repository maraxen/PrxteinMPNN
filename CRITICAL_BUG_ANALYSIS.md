# Critical Bug Analysis: Alphabet Mismatch in PrxteinMPNN Validation

**Date**: 2025-11-07
**Branch**: debug
**Status**: 🔴 **CRITICAL BUG IDENTIFIED AND PARTIALLY FIXED**

---

## 🎯 Executive Summary

Identified and partially fixed a critical bug in the validation comparison between PrxteinMPNN and ColabDesign that was causing artificially low correlation metrics and misleading validation results.

### Root Cause

**Alphabet Mismatch in Logits Comparison**: The comparison notebook was directly comparing logits from PrxteinMPNN (in MPNN alphabet order) with logits from ColabDesign (in AF alphabet order) without performing the necessary alphabet conversion.

### Impact

- **Before Fix**: Pearson correlation = 0.17, Cosine similarity = 0.24
- **After Alphabet Fix**: Pearson correlation = 0.62, Cosine similarity = 0.65
- **Improvement**: +264% correlation, +171% cosine similarity

### Remaining Issues

Even after alphabet conversion, correlation is still below the expected >0.9 threshold, indicating additional implementation differences that need investigation.

---

## 📊 Bug Details

### The Two Alphabet Systems

Both implementations use different amino acid orderings:

```python
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"  # ProteinMPNN native order
AF_ALPHABET   = "ARNDCQEGHILKMFPSTWYVX"  # AlphaFold order
```

### How Each Implementation Handles Alphabets

**PrxteinMPNN**:
- Parses PDB files using AlphaFold alphabet (via `restype_order` from residue_constants)
- Converts to MPNN alphabet during parsing: `af_to_mpnn()` (line 139 in mappings.py)
- Stores sequences in MPNN alphabet internally
- Returns logits in MPNN alphabet (column 0 = 'A', column 1 = 'C', etc.)

**ColabDesign**:
- Parses PDB files using AlphaFold alphabet (via AlphaFold's `prep_pdb`)
- Stores sequences in AF alphabet
- Converts inputs from AF→MPNN before passing to model (line 252/285 in model.py)
- **Converts outputs from MPNN→AF** after model returns (line 255-256/288-289)
- Returns logits in AF alphabet (column 0 = 'A', column 1 = 'R', etc.)

### The Bug in Validation Code

The comparison notebook (`prxteinmpnn_vs_colabdesign_comparison.ipynb`) was:

1. Getting logits from PrxteinMPNN (MPNN alphabet order)
2. Getting logits from ColabDesign (AF alphabet order)
3. **Directly comparing them without conversion** ❌

This is like comparing:
```
PrxteinMPNN column 1 = logits for 'C'
ColabDesign column 1 = logits for 'R'
```

Obviously these will not correlate!

---

## 🔬 Validation Results

### Input Validation

Verified that both implementations receive identical inputs:

✅ **Identical**:
- Backbone atom coordinates (N, CA, C, O) - match to <0.000001 Å
- Mask values
- Residue indices
- Chain indices

❌ **Different** (but expected):
- Coordinate array shapes (PrxteinMPNN: 76×37×3, ColabDesign: 76×4×3)
  - Not an issue: PrxteinMPNN only uses first 4 atoms internally
- Sequence encoding alphabet (MPNN vs AF)
  - Not an issue: can be converted

### Alphabet Conversion Test Results

**Test**: Unconditional logits on 1UBQ.pdb (76 residues)

| Metric | Without Conversion (BUGGY) | With Conversion (FIXED) | Expected | Status |
|--------|----------------------------|-------------------------|----------|--------|
| **Pearson Correlation** | 0.1699 | 0.6183 | >0.90 | ⚠️ Improved but still low |
| **Cosine Similarity** | 0.2435 | 0.6464 | >0.90 | ⚠️ Improved but still low |
| **Prediction Agreement** | 14.5% | 35.5% | >80% | ⚠️ Improved but still low |
| **PrxteinMPNN Recovery** | 21.1% | 21.1% | 35-65% | ❌ Below expected |
| **ColabDesign Recovery** | 55.3% | 55.3% | 35-65% | ✅ Within range |

**Logits Range Comparison**:
- PrxteinMPNN: [-2.645, 2.297] (range: 4.942)
- ColabDesign: [-2.690, 5.277] (range: 7.967)

ColabDesign's logits have **61% wider range**, suggesting more confident predictions.

---

## 🐛 Remaining Issues to Investigate

The alphabet fix improved correlation significantly but not to the expected >0.9 level. This suggests additional differences:

### Priority 1: Model Weights Verification

**Hypothesis**: The models might be loading different weights.

**Evidence**:
- Logits range differs significantly
- ColabDesign has much better recovery (55.3% vs 21.1%)
- Logits correlation is only 0.62 after alphabet fix

**Investigation needed**:
1. Verify both models load from same source weights file
2. Compare a few weight matrices directly (e.g., encoder layer 1 weights)
3. Check if weight loading/conversion has any bugs

**How to test**:
```python
# Load both models
prx_model = load_prxteinmpnn(...)
colab_model = mk_mpnn_model(...)

# Compare specific weights
# Need to find equivalent layers in both implementations
```

### Priority 2: Feature Preprocessing

**Hypothesis**: Edge features or node features might be computed differently.

**Evidence**:
- Coordinates are identical
- But final logits still differ after alphabet correction

**Investigation needed**:
1. Extract and compare edge features (RBF, distances)
2. Compare node features after encoder
3. Check positional encodings

**Files to check**:
- `src/prxteinmpnn/model/features.py` - ProteinFeatures class
- `/tmp/ColabDesign/colabdesign/mpnn/modules.py` - Features computation

### Priority 3: Encoder/Decoder Architecture

**Hypothesis**: Subtle differences in attention mechanisms or layer normalization.

**Investigation needed**:
1. Compare encoder outputs position-by-position
2. Compare decoder context preparation
3. Verify attention mask construction

**Files to check**:
- `src/prxteinmpnn/model/encoder.py`
- `src/prxteinmpnn/model/decoder.py`
- `/tmp/ColabDesign/colabdesign/mpnn/modules.py`

---

## 🔧 How to Apply the Alphabet Conversion Fix

### For Logits Comparison

When comparing logits from ColabDesign to PrxteinMPNN:

```python
import numpy as np

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

def af_logits_to_mpnn(logits_af):
    """
    Convert logits from AF alphabet order to MPNN alphabet order.

    Args:
        logits_af: Array of shape [..., 21] with logits in AF alphabet order

    Returns:
        logits_mpnn: Array of shape [..., 21] with logits in MPNN alphabet order
    """
    # For each position in MPNN alphabet, find where that AA is in AF alphabet
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    return logits_af[..., perm]

# Usage:
colab_logits_af = colab_model.get_unconditional_logits(...)  # AF alphabet order
colab_logits_mpnn = af_logits_to_mpnn(colab_logits_af)  # Convert to MPNN order
# Now can compare with PrxteinMPNN logits
correlation = np.corrcoef(prxtein_logits.flatten(), colab_logits_mpnn.flatten())[0, 1]
```

### For Sequence Comparison

When comparing sequences:

```python
def af_seq_to_mpnn(seq_af):
    """Convert sequence from AF alphabet indices to MPNN alphabet indices."""
    return np.array([MPNN_ALPHABET.index(AF_ALPHABET[idx]) for idx in seq_af])

# Usage:
prx_seq = protein.aatype  # Already in MPNN alphabet
colab_seq_af = colab_model._inputs['S']  # In AF alphabet
colab_seq_mpnn = af_seq_to_mpnn(colab_seq_af)  # Convert to MPNN
# Now can compare
match = (prx_seq == colab_seq_mpnn).sum() / len(prx_seq)
```

---

## 📌 Files Modified/Created

### Created Files:
1. `debug_sequence_parsing.py` - Initial debugging script
2. `test_alphabet_conversion_fix.py` - Alphabet conversion validation
3. `test_input_validation.py` - Input comparison between implementations
4. `CRITICAL_BUG_ANALYSIS.md` - This document

### Files to Fix:
1. `notebooks/prxteinmpnn_vs_colabdesign_comparison.ipynb` - Needs alphabet conversion
2. `tests/validation/test_colabdesign_comparison.py` - Needs alphabet conversion
3. Any other validation/comparison code

---

## ✅ Next Steps

1. **Immediate**: Update all comparison code to use proper alphabet conversion
2. **Priority 1**: Investigate weight loading to ensure both models use identical weights
3. **Priority 2**: Compare intermediate features (edge features, encoder outputs)
4. **Priority 3**: Once correlation reaches >0.9, re-run full validation suite

---

## 📞 Key Findings for Next Developer

1. **The alphabet mismatch was real and significant** - improved correlation by 264%
2. **But it's not the only issue** - still at 0.62 correlation vs expected >0.9
3. **Inputs are verified identical** - problem is in model internals
4. **ColabDesign performs much better** - suggests PrxteinMPNN might have additional bugs
5. **Next focus should be on weights** - most likely source of remaining difference

---

## 🔍 Debugging Tools Created

### Quick Test Command:
```bash
uv run python test_alphabet_conversion_fix.py
```

This will show before/after metrics with alphabet conversion.

### Input Validation Command:
```bash
uv run python test_input_validation.py
```

This will verify all inputs match between implementations.

---

**Remember**: Always convert logits to the same alphabet before comparing!
