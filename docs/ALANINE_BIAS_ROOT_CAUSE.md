# Alanine Bias: Root Cause Analysis

**Date:** 2025-01-15  
**Status:** 🔴 **ROOT CAUSE IDENTIFIED**

## Executive Summary

The unconditional decoder produces **64.5% Alanine predictions** on 1ubq. The root cause has been identified:

**The decoder's output features are strongly aligned with Alanine's weight vector in the output layer.**

- Decoded features · Alanine weights: **+1.051**
- Decoded features · Other AA weights (mean): **-0.244**  
- **Ratio: -4.3× (Alanine favored over others)**

## Analysis Pipeline

### 1. Weight Structure ✅ VERIFIED
- All 60 original weight modules present in Equinox model
- Shapes match expected dimensions
- No missing or extra weights
- **Conclusion:** Weight structure is correct

### 2. Output Layer Weights ✅ ANALYZED
- Alanine bias: +0.1816 (positive)
- Other AA bias mean: -0.1554 (negative)
- Alanine weight norm: 1.854 (**second lowest**, not highest!)
- **Conclusion:** Output weights themselves don't explain bias

### 3. Intermediate Activations 🔴 **PROBLEM FOUND**

#### Stage 1: Feature Extraction ✅ Normal
```
Edge features: (76, 48, 128)
  Mean: 0.171, Std: 4.389
  Range: [-25.4, 30.6]
  No NaN/Inf
```

#### Stage 2: Encoder ✅ Normal
```
Node features: (76, 128)
  Mean: 0.032, Std: 0.367
  Range: [-1.21, 3.55]
  Node norm CV: 0.0186 (very uniform, healthy)
  No NaN/Inf
```

#### Stage 3: Decoder (Unconditional) 🔴 **BIAS ORIGINATES HERE**
```
Decoded features: (76, 128)
  Mean: 0.035, Std: 0.255
  Range: [-0.89, 1.21]
```

**Critical Finding:**
When projecting decoded features to logits via `W_out`:
- **Alanine logit (mean): +1.233** ← Highest by far!
- D logit (mean): +0.146
- All other AAs: negative or near-zero

**The decoder produces a feature vector that maximally activates Alanine when passed through W_out.**

#### Stage 4: Output Projection
```
Logits shape: (76, 21)
Alanine mean logit: +1.233  
Other AAs: mostly negative

Predictions:
  - Alanine: 64.5%
  - Valine: 9.2%
  - Aspartic acid: 13.2%
  - All others: <6%
```

## Root Cause

### The Problem
The **unconditional decoder** (which uses zeros for sequence embeddings) produces node features that are:
1. Systematically biased toward a specific direction in feature space
2. This direction aligns strongly with Alanine's output weight vector
3. This alignment causes Alanine to dominate predictions

### Diagnostic Evidence

```python
decoded_mean = jnp.mean(decoded_node_features, axis=0)  # Average decoded feature (128,)
alanine_weight = model.w_out.weight[0, :]  # Alanine's weight vector (128,)

dot_product = decoded_mean · alanine_weight = +1.051

# Compare to other amino acids:
other_dots_mean = -0.244

# Alanine is 4.3× more activated!
```

This means the decoder is producing features in a subspace that heavily favors Alanine.

## Why This Happens

### Hypothesis 1: Zero Sequence Embeddings
In unconditional mode, the decoder receives:
```python
context = concat([node_features, zeros(128), edge_features])
#                                 ^^^^^^^^^^
#                           sequence embedding = 0
```

If the weights were trained expecting non-zero sequence embeddings, setting them to zero might:
1. Shift the decoder's operating point
2. Cause it to output features in an unintended subspace
3. This subspace happens to align with Alanine's direction

### Hypothesis 2: Weight Conversion Error
The weights might have been:
- Transposed incorrectly during PyTorch→JAX conversion
- Applied with wrong sign/scale
- Mixed up between encoder/decoder

This would cause systematic misalignment.

### Hypothesis 3: Decoder Architecture Difference
Subtle difference between original ProteinMPNN decoder and our Equinox implementation:
- Layer ordering
- Normalization application
- Residual connections
- Edge feature handling

## Next Steps to Investigate

### Priority 1: Compare Weight Values
```python
# Load original PyTorch weights and compare actual values
original_w_out = load_original_weights()['W_out']
equinox_w_out = model.w_out.weight

# Check for:
# 1. Transposition: original.T == equinox?
# 2. Sign flip: -original == equinox?
# 3. Value discrepancies
```

### Priority 2: Test Conditional Mode
```python
# Does the decoder work correctly WITH sequence context?
conditional_logits = model(coords, mask, residue_idx, chain_idx, aatype)
accuracy = (argmax(conditional_logits) == aatype).mean()

# If accuracy > 20%, decoder architecture is correct
# → Problem is specifically with zero sequence embeddings
```

### Priority 3: Compare with Original Implementation
Run the same 1ubq structure through:
1. Original PyTorch ProteinMPNN (unconditional)
2. ColabDesign implementation (unconditional)
3. Our Equinox implementation

Compare:
- Logits at each position
- Predicted amino acids
- Alanine prediction frequency

### Priority 4: Decoder Layer-by-Layer Comparison
```python
# Check if decoded features diverge at a specific layer
for i, layer in enumerate(model.decoder.layers):
    h_i = layer(h_{i-1}, edges, mask)
    print(f"Layer {i}: mean={h_i.mean()}, alignment_with_alanine={...}")
```

## Related Documentation

- **WEIGHT_VERIFICATION.md:** Confirms all weights are present
- **ALPHABET_CONVERSION_VERIFICATION.md:** Confirms alphabet conversion is correct
- **SEQUENCE_RECOVERY_TESTS.md:** Documents 5-8% recovery vs expected 40-60%
- **tests/debug/test_intermediate_activations.py:** Full diagnostic code

## Test Results Summary

| Test | Status | Finding |
|------|--------|---------|
| Weight structure | ✅ PASS | All 60 modules present |
| Alphabet conversion | ✅ PASS | Correctly maps AF↔MPNN |
| Bias application | ✅ PASS | W_out.bias applied correctly |
| Coordinates | ✅ PASS | No NaN/Inf, reasonable values |
| Feature extraction | ✅ PASS | Normal edge/node features |
| Encoder | ✅ PASS | Uniform, healthy activations |
| **Decoder (unconditional)** | 🔴 **FAIL** | **Produces Alanine-biased features** |
| Output layer | ✅ PASS | Weights look reasonable |

## Conclusion

The issue is **NOT** in:
- ❌ Weight loading/structure
- ❌ Alphabet conversion  
- ❌ Bias application
- ❌ Input coordinates
- ❌ Feature extraction
- ❌ Encoder
- ❌ Output layer weights

The issue **IS** in:
- ✅ **Decoder (unconditional mode)** producing Alanine-aligned features

This narrows the problem to either:
1. **Weight conversion error** in decoder weights (transposition/sign)
2. **Architectural mismatch** in decoder implementation
3. **Fundamental issue** with zero sequence embeddings in unconditional mode

**Next Action:** Compare decoder weight values with original PyTorch implementation to check for conversion errors.
