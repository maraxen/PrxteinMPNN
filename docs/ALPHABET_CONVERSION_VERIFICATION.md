# Alphabet Conversion Verification

## Summary

Verified that alphabet conversion between AlphaFold (AF) and ProteinMPNN (MPNN) amino acid orderings is working correctly. **This is NOT the cause of low recovery rates.**

## Amino Acid Alphabets

- **AF_ALPHABET**: `"ARNDCQEGHILKMFPSTWYVX"`
- **MPNN_ALPHABET**: `"ACDEFGHIKLMNPQRSTVWYX"`

## Verification Results

### Input Processing ✓
Parsing correctly converts sequences from AF→MPNN order:

```python
# Example: 1ubq starts with "MQIFVKTLTG"
MET -> resname_to_idx[MET]=12 (AF) -> af_to_mpnn -> 10 (MPNN) ✓
GLN -> resname_to_idx[GLN]=5  (AF) -> af_to_mpnn -> 13 (MPNN) ✓
# etc.
```

Verified with:
```python
from prxteinmpnn.io.parsing import parse_input
protein_tuple = next(parse_input('tests/data/1ubq.pdb'))
aatype = protein_tuple[3]  # [10, 13, 7, 4, 17, 8, 16, 9, 16, 5, ...]
# Correctly maps to "MQIFVKTLTG" in MPNN alphabet
```

### Model Processing ✓
- Model weights were trained with MPNN alphabet
- Logits output is in MPNN order  
- Predictions compared to ground truth (also MPNN order) - comparison is valid

### Code Locations

**Conversion functions:**
- `src/prxteinmpnn/io/parsing/mappings.py`: `af_to_mpnn()`, `mpnn_to_af()`
- `src/prxteinmpnn/utils/aa_convert.py`: JAX versions of conversion functions

**Usage in parsing:**
- `residue_names_to_aatype()` calls `af_to_mpnn()` automatically
- Used by: `biotite.py`, `mdtraj.py`, `mdcath.py`

## Conclusion

The alphabet conversion is implemented correctly and is NOT causing the low recovery rates (5-8% vs expected 40-60%). The issue lies elsewhere, most likely in:

1. **Weight conversion from PyTorch to Equinox** - Potential mismatch in how weights were converted
2. **Architecture differences** - Subtle implementation differences vs original ProteinMPNN
3. **Missing normalization** - Possible missing layer norm or other preprocessing step

## Next Steps

1. Compare Equinox model architecture line-by-line with original ProteinMPNN
2. Verify weight loading/conversion process
3. Test with original ProteinMPNN checkpoints to confirm weights are compatible

---
**Date**: November 7, 2025
**Status**: ✅ Alphabet conversion verified correct
**Finding**: Alphabet conversion is NOT the root cause
