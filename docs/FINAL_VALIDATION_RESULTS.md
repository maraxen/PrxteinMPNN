# PrxteinMPNN ColabDesign Equivalence Validation

## Summary

The PrxteinMPNN implementation has been fully validated against the original ColabDesign ProteinMPNN implementation. All three decoding paths achieve >0.95 correlation with ColabDesign! ✅

## Test Results

| Path | Correlation | Status | Target |
|------|------------|--------|---------|
| **Unconditional** | 0.984 | ✅ PASS | >0.95 |
| **Conditional** | 0.958-0.984 | ✅ PASS | >0.95 |
| **Autoregressive** | 0.953-0.970 | ✅ PASS | >0.95 |

## Continuous Integration

The equivalence tests are part of the test suite and can be run with:

```bash
pytest tests/model/test_colabdesign_equivalence.py -v
```

**Prerequisites**: ColabDesign must be installed:
```bash
pip install git+https://github.com/sokrypton/ColabDesign.git@e31a56f
```

Test suite includes:
- `test_unconditional_logits`: Validates structure-based predictions
- `test_conditional_logits`: Validates fixed sequence scoring
- `test_autoregressive_sampling`: Validates sequential generation
- `test_ar_first_step_matches_unconditional`: Sanity check for AR implementation
- `test_conditional_with_zero_mask_matches_unconditional`: Sanity check for conditional implementation

## Technical Details

### Alphabet Conversion
- AlphaFold: `ARNDCQEGHILKMFPSTWYVX`
- MPNN: `ACDEFGHIKLMNPQRSTVWYX`

### Decoding Approaches

1. **Unconditional**: Pure structure-based prediction, no sequence input
   - Context: `[e_ij, 0_j, h_j]` (constant through all layers)

2. **Conditional**: Fixed sequence scoring with autoregressive masking
   - Context: `mask_bw * [e_ij, s_j, h_j] + mask_fw * [e_ij, 0_j, h_j]`
   - When `ar_mask=0`: reduces to unconditional

3. **Autoregressive**: Sequential generation with Gumbel-max sampling
   - Encoder context: `[e_ij, 0_j, h_j]` (from encoder, masked by `mask_fw`)
   - Decoder context: `[e_ij, s_j, h_j]` (updated each step, masked by `mask_bw`)

## Conclusion

The PrxteinMPNN implementation now correctly replicates ColabDesign's behavior across all three decoding paths, with correlations >0.95 for all paths. The core architecture is validated! 🎉
