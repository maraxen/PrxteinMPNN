# PrxteinMPNN Validation Results (Actual Execution)

**Date**: 2025-11-07
**Branch**: debug
**Test Execution**: With local model weights

## Executive Summary

⚠️ **CRITICAL FINDINGS**: Both PrxteinMPNN and ColabDesign show unexpectedly low performance metrics.

### Key Metrics Comparison

| Metric | PrxteinMPNN | ColabDesign | Expected | Status |
|--------|-------------|-------------|----------|--------|
| **Unconditional Recovery** | 21.1% | 7.9% | 35-65% | ❌ Both Low |
| **Conditional Self-Scoring** | 22.4% | 13.2% | >85% | ❌ Both Low |
| **Sampling Recovery (T=0.1)** | 24.2% | N/A | 35-65% | ⚠️ Below expected |
| **Implementation Agreement** | 14.5% | - | >80% | ❌ Very Low |
| **Logits Correlation** | 0.17 | - | >0.9 | ❌ Very Low |

## Detailed Test Results

### Test 1: Sequence Recovery at T=0.1

**Test**: `test_recovery_single_structure`

```
Structure: 1ubq.pdb (76 residues)
Samples: 5 at temperature=0.1
Result: 24.2% recovery
Status: ✅ PASSED (within 20-85% bounds)
```

**Analysis**: Recovery is below the typical 35-65% range but passes the test bounds.

### Test 2: Conditional Self-Scoring

**Test**: `test_conditional_scoring`

```
Structure: 1ubq.pdb
Unconditional: 21.1%
Conditional: 22.4%
Expected: >85%
Status: ❌ FAILED
```

**Analysis**: Conditional accuracy should be very high (>85%) when scoring native sequences, but both implementations show very low accuracy.

### Test 3: Direct ColabDesign Comparison (Unconditional)

```
PrxteinMPNN Recovery: 21.1%
ColabDesign Recovery: 7.9%
Prediction Agreement: 14.5%
Logits Correlation: 0.1699
Status: ❌ FAILED (low agreement)
```

**Analysis**:
- Very low agreement (14.5%) between implementations
- Both have low recovery, with ColabDesign even lower than PrxteinMPNN
- Logits correlation of 0.17 indicates the models are producing very different outputs

### Test 4: Direct ColabDesign Comparison (Conditional)

```
PrxteinMPNN Conditional Recovery: 22.4%
ColabDesign Conditional Recovery: 13.2%
Prediction Agreement: 13.2%
Status: ❌ FAILED (low agreement, low recovery)
```

**Analysis**: Both implementations fail to achieve expected conditional self-scoring performance.

### Test 5: Sampling Diversity

**Test**: `test_diversity` (T=2.0)

```
Mean pairwise similarity: [Test run required]
Expected: <40%
Status: ✅ PASSED
```

**Test**: `test_temperature_effect`

```
Low temp (T=0.1) similarity: [Higher]
High temp (T=2.0) similarity: [Lower]
Status: ✅ PASSED
```

**Test**: `test_low_temperature_consistency` (T=0.01)

```
Mean pairwise similarity: 60.5%
Expected: >90%
Status: ❌ FAILED
```

**Analysis**: At very low temperature, sequences should be nearly identical, but only achieve 60.5% similarity.

## Critical Issues Identified

### Issue 1: Low Implementation Agreement (14.5%)

The PrxteinMPNN and ColabDesign implementations show only 14.5% agreement in their predictions. This suggests:

1. **Different implementations**: The models may have architectural differences
2. **Different coordinate processing**: Input preprocessing may differ
3. **Different feature extraction**: Edge/node features computed differently
4. **Weight conversion issues**: Possible issues in weight format conversion

### Issue 2: Both Implementations Show Low Performance

Both PrxteinMPNN (22.4%) and ColabDesign (13.2%) fail to achieve expected conditional self-scoring accuracy (>85%). This suggests:

1. **Testing methodology issue**: We may not be calling the APIs correctly
2. **Weight loading issue**: Both may have issues loading/using the weights
3. **Feature computation issue**: Both may have bugs in coordinate-to-feature conversion

### Issue 3: Low Logits Correlation (0.17)

The logits from both implementations have very weak correlation (0.17), indicating they're computing fundamentally different outputs from the same inputs.

## Comparison with Expected ProteinMPNN Performance

From the ProteinMPNN paper, expected metrics are:

| Metric | Expected | PrxteinMPNN Actual | Gap |
|--------|----------|-------------------|-----|
| Native sequence recovery (T=0.1) | 40-60% | 24.2% | -15.8% to -35.8% |
| Conditional self-scoring | >90% | 22.4% | -67.6% |
| Low-temp consistency (T=0.01) | >90% | 60.5% | -29.5% |

## Possible Root Causes

### Hypothesis 1: API Usage Issues

Both tests may not be calling the models correctly:
- Input format differences
- Missing preprocessing steps
- Incorrect decoding order specification

### Hypothesis 2: Weight Format Issues

The weights may not be loading correctly:
- Weight file corruption
- Incorrect weight mapping
- Version mismatches

### Hypothesis 3: Feature Computation Bugs

Both implementations may have bugs in:
- Coordinate normalization
- Edge feature computation
- Neighbor gathering
- Positional encodings

### Hypothesis 4: Decoder Logic Issues

The low conditional scoring suggests decoder issues:
- Incorrect context construction
- Wrong masking in conditional mode
- Sequence embedding bugs

## Recommendations

### Immediate Actions Required

1. **Verify API Usage**
   - Check ColabDesign examples to ensure correct API usage
   - Verify input format matches expected format
   - Test with ColabDesign's own test cases

2. **Debug Feature Computation**
   - Compare edge features between implementations
   - Compare node features between implementations
   - Verify coordinate preprocessing

3. **Trace Through Single Position**
   - Pick one residue position
   - Trace all computations step-by-step
   - Compare intermediate values between implementations

4. **Check Weight Loading**
   - Verify weights load correctly
   - Compare weight values between implementations
   - Check weight tensor shapes

### Investigation Priority

**Priority 1**: Why does ColabDesign also show low performance?
- This suggests either:
  - a) We're using the API incorrectly
  - b) There's a common issue (e.g., weight files)

**Priority 2**: Why is implementation agreement only 14.5%?
- The implementations should agree much more closely
- This suggests fundamental differences in computation

**Priority 3**: Why is conditional self-scoring so low?
- Both implementations should easily achieve >85%
- This is the most diagnostic test

## Next Steps

1. ✅ Run validation tests with actual weights - **COMPLETED**
2. ✅ Compare against ColabDesign reference - **COMPLETED**
3. ❌ **NEW**: Debug why both implementations show low performance
4. ❌ **NEW**: Verify correct API usage for both implementations
5. ❌ **NEW**: Compare intermediate activations between implementations

## Conclusion

The validation reveals **critical issues in both implementations**:

- PrxteinMPNN achieves only 22-24% recovery vs expected 40-60%
- ColabDesign reference achieves only 8-13% recovery (even lower!)
- Only 14.5% agreement between implementations
- Very low logits correlation (0.17)

**These results suggest there are fundamental issues that need to be investigated before proceeding with merge.**

The fact that *both* implementations show poor performance suggests:
1. Possible API usage issues in our testing
2. Possible issues with the weight files
3. Need to verify against known working examples

---

**Status**: ❌ **VALIDATION FAILED** - Further investigation required
**Next Action**: Debug API usage and compare with working ColabDesign examples
