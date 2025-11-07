# PrxteinMPNN Validation: Critical Issues & Debugging Handoff

**Date**: 2025-11-07
**Session**: claude/prxteinmpnn-validation-cleanup-011CUuBUPG2G7Z9ArK9tw5Cv
**Status**: 🔴 **CRITICAL ISSUES FOUND** - Back to debugging phase

---

## 🎯 Executive Summary

Comprehensive validation tests revealed **fundamental implementation issues** in PrxteinMPNN that prevent it from matching the ColabDesign (reference) implementation. The model produces drastically different logits and achieves very poor sequence recovery (5.7% vs expected 35-65%).

### Key Findings:
1. **Sequence Mismatch**: Native sequences differ between implementations despite reading same PDB
2. **Logits Pattern Mismatch**: PrxteinMPNN produces low-confidence, narrow-range logits vs ColabDesign's high-confidence outputs
3. **Recovery Rate**: PrxteinMPNN achieves only 5.7% recovery (ColabDesign: 16.2%, both far below 35-65% expected)
4. **Logits Similarity**: Very low Pearson correlation and cosine similarity between implementations

---

## 📊 Observed Problems

### Problem 1: Sequence Indexing Mismatch

**Observation**: Reading the same PDB file (1UBQ) produces different sequence arrays:

```
PrxteinMPNN:  [10 13  7  4 17  8 16  9 16  5  8 16  7 16  9  3 17  3 12 15]
ColabDesign:  [12  5  9 13 19 11 16 10 16  7 11 16  9 16 10  6 19  6 14 15]
```

**Implications**:
- Different amino acid indexing system being used
- Possible issues in PDB parsing
- Could explain all downstream differences

**Evidence**:
- Both implementations claim to parse 1UBQ.pdb
- Both report 76 residues
- But integer indices are completely different

### Problem 2: Logits Distribution Mismatch

**Observation**: Heatmap visualization shows stark differences:

| Feature | PrxteinMPNN | ColabDesign |
|---------|-------------|-------------|
| **Contrast** | Low, muted | High, sharp |
| **Value Range** | Clustered around 0 (±2) | Full range (-3 to +3) |
| **Extreme Values** | Very few | Many dark red/blue cells |
| **Implied Confidence** | Low | High |

**Implications**:
- PrxteinMPNN logits are "flat" (uncertain predictions)
- ColabDesign logits are "sharp" (confident predictions)
- Suggests fundamental issue in encoder/decoder or logit computation

**Visual Evidence**: See notebook heatmaps in cell outputs

### Problem 3: Poor Sequence Recovery

**Observation**: Both implementations fail to achieve expected performance:

```
Expected:        35-65% at T=0.1
PrxteinMPNN:     5.7% ± 2.2%
ColabDesign:     16.2% ± 5.6%
```

**Implications**:
- PrxteinMPNN is worse than ColabDesign
- ColabDesign itself may have API usage issues
- Need to verify against known working examples

---

## 🔍 Investigation Findings

### Amino Acid Alphabet Conversion

Both implementations use different alphabet orderings:

```python
# ColabDesign (line 312-314 of model.py):
mpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'  # ProteinMPNN native order
af_alphabet =   'ARNDCQEGHILKMFPSTWYVX'  # AlphaFold order
```

**PrxteinMPNN Implementation**:
- ✅ Has conversion functions: `af_to_mpnn()` and `mpnn_to_af()` in:
  - `src/prxteinmpnn/utils/aa_convert.py`
  - `src/prxteinmpnn/io/parsing/mappings.py`
- ✅ Uses conversion in parsing: `residue_names_to_aatype()` (line 139 of mappings.py)
- ❓ **NEEDS VERIFICATION**: Is conversion being applied consistently throughout?

**ColabDesign Implementation**:
- Uses `_aa_convert()` function to convert between alphabets
- Applies conversion in `_score()` method before model.score() call
- Converts logits back after model output

### Unconditional Decoding Flow

**ColabDesign Flow** (`get_unconditional_logits`):
```python
def get_unconditional_logits(self, **kwargs):
    L = self._inputs["X"].shape[0]
    kwargs["ar_mask"] = np.zeros((L,L))  # No autoregressive masking
    return self.score(**kwargs)["logits"]
```

**Key Details**:
1. Sets `ar_mask = np.zeros((L,L))` → all positions can attend to all positions
2. Calls `score()` which:
   - Converts input sequence to MPNN alphabet
   - Generates random decoding order
   - Calls `model.score()`
   - Converts logits back to AF alphabet

**PrxteinMPNN Flow** (`_call_unconditional`):
```python
def _call_unconditional(self, edge_features, neighbor_indices, mask, ...):
    node_features, processed_edge_features = self.encoder(
        edge_features, neighbor_indices, mask
    )
    decoded_node_features = self.decoder(
        node_features, processed_edge_features, mask
    )
    logits = jax.vmap(self.w_out)(decoded_node_features)
    return dummy_seq, logits
```

**Key Differences to Investigate**:
1. ❓ Does PrxteinMPNN decoder use correct unconditional context?
2. ❓ Is the decoder called correctly (no AR mask in unconditional)?
3. ❓ Are edge features computed identically?

---

## 🐛 Critical Debugging Steps

### Step 1: Verify Sequence Parsing Consistency ⚠️ **HIGH PRIORITY**

**Goal**: Ensure both implementations read the same sequence from 1UBQ.pdb

**Tasks**:
1. Add debug logging to print:
   - Raw residue names from PDB
   - Indices before conversion
   - Indices after af_to_mpnn conversion
2. Compare with ColabDesign's parsed sequence
3. Verify conversion tables match ColabDesign exactly

**Expected Outcome**: Both implementations should have identical integer sequences (in MPNN alphabet)

**Files to Check**:
- `src/prxteinmpnn/io/parsing/biotite.py` (line 139)
- `src/prxteinmpnn/io/parsing/mappings.py` (af_to_mpnn, residue_names_to_aatype)
- `/tmp/ColabDesign/colabdesign/mpnn/model.py` (_aa_convert)

### Step 2: Trace Unconditional Decoding Path ⚠️ **HIGH PRIORITY**

**Goal**: Ensure unconditional decoding matches ColabDesign's flow

**Tasks**:
1. Compare encoder outputs (node features, edge features):
   ```python
   # Add after encoder call:
   print(f"Node features stats: min={node_features.min():.4f}, max={node_features.max():.4f}, mean={node_features.mean():.4f}")
   print(f"Edge features stats: min={processed_edge_features.min():.4f}, max={processed_edge_features.max():.4f}")
   ```

2. Verify decoder context for unconditional mode:
   - Check decoder.py line 237-248 (unconditional context)
   - Should be: `[h_i, 0, e_ij]` (NOT using sequence embeddings)

3. Compare with ColabDesign's decoder context

4. Check if AR mask is properly disabled in unconditional mode

**Files to Check**:
- `src/prxteinmpnn/model/decoder.py` (lines 237-258)
- `src/prxteinmpnn/model/mpnn.py` (_call_unconditional, lines 130-187)

### Step 3: Compare Edge Feature Computation ⚠️ **MEDIUM PRIORITY**

**Goal**: Verify edge features are computed identically

**Tasks**:
1. Compare RBF (Radial Basis Function) encoding parameters
2. Check distance calculations (CA-CA distances)
3. Verify positional encodings
4. Compare edge message passing in encoder

**Files to Check**:
- `src/prxteinmpnn/model/features.py`
- `src/prxteinmpnn/model/encoder.py`
- ColabDesign equivalent files

### Step 4: Verify Model Architecture ⚠️ **MEDIUM PRIORITY**

**Goal**: Ensure model layers match reference implementation

**Tasks**:
1. Compare layer dimensions:
   - Node features: 128
   - Edge features: 128
   - Hidden features: 128
   - Num encoder layers: 3
   - Num decoder layers: 3

2. Verify activation functions (ReLU, etc.)

3. Check normalization (if any)

4. Compare attention mechanisms

### Step 5: Test with Known Working Example ⚠️ **LOW PRIORITY**

**Goal**: Verify ColabDesign API usage is correct

**Tasks**:
1. Find existing ColabDesign examples/tutorials
2. Run them to verify expected performance
3. Compare API usage with our notebook
4. Adjust our usage if needed

---

## 📁 Artifacts Created

### Test Files
- `tests/validation/test_colabdesign_comparison.py` - Direct comparison test
- `tests/validation/test_extensive_recovery.py` - Recovery rate tests
- `tests/validation/test_conditional_accuracy.py` - Conditional scoring tests
- `tests/validation/test_sampling_diversity.py` - Temperature sweep tests

### Notebooks & Code
- `notebooks/prxteinmpnn_vs_colabdesign_comparison.ipynb` - Comprehensive Colab notebook
  - Enhanced with violin plots
  - Chemical property-based logits visualization
  - Multi-structure testing framework (6 structures, 10-76 residues)
  - Cosine similarity + Pearson correlation metrics

- `notebooks/multi_structure_comparison.py` - Multi-structure test code

### Documentation
- `VALIDATION_RESULTS.md` - Detailed validation findings
- `DEBUGGING_HANDOFF.md` - This document

---

## 🎨 Visualization Enhancements

### Added Metrics:
1. **Cosine Similarity**: Angular similarity between logit vectors
   - Less sensitive to magnitude than Pearson correlation
   - Provides complementary similarity measure

2. **Violin Plots**: Distribution visualization for:
   - Unconditional recovery rates
   - Conditional self-scoring accuracy
   - Sampling diversity across temperatures

3. **Chemical Property Coloring**: Amino acid-based scatter plot coloring:
   - Hydrophobic (brown): A, G, I, L, M, V
   - Aromatic (purple): F, W, Y
   - Polar (green): C, N, Q, S, T
   - Positively charged (blue): H, K, R
   - Negatively charged (red): D, E
   - Proline (gold): P

---

## 🔧 Code Investigation Strategy

### Recommended Approach:

1. **Start with sequence parsing** (highest impact, easiest to debug)
   - If sequences don't match, nothing else will work
   - Should be quick to verify

2. **Move to unconditional decoding** (core functionality)
   - Compare intermediate outputs step-by-step
   - Use small test cases for faster iteration

3. **Then edge features** (if above checks pass)
   - More complex but well-defined

4. **Finally architecture verification** (if all else fails)
   - Most time-consuming
   - Should be last resort

### Debugging Tools:

```python
# Add strategic print statements:
def debug_tensor(name, tensor):
    print(f"{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  min: {tensor.min():.4f}")
    print(f"  max: {tensor.max():.4f}")
    print(f"  mean: {tensor.mean():.4f}")
    print(f"  std: {tensor.std():.4f}")

# Use at key points:
debug_tensor("encoder_node_features", node_features)
debug_tensor("decoder_logits", logits)
```

### Comparison Strategy:

```python
# Direct numerical comparison:
def compare_with_colabdesign(prxtein_output, colab_output, name="output"):
    diff = np.abs(prxtein_output - colab_output)
    print(f"\n{name} Comparison:")
    print(f"  Max absolute diff: {diff.max():.6f}")
    print(f"  Mean absolute diff: {diff.mean():.6f}")
    print(f"  Correlation: {np.corrcoef(prxtein_output.flatten(), colab_output.flatten())[0,1]:.6f}")

    # Check if they're within numerical tolerance
    if np.allclose(prxtein_output, colab_output, rtol=1e-5, atol=1e-6):
        print(f"  ✅ Outputs match within tolerance")
    else:
        print(f"  ❌ Outputs differ significantly")
```

---

## 📌 Important References

### ColabDesign Key Files:
- `/tmp/ColabDesign/colabdesign/mpnn/model.py`
  - Line 226-229: `get_unconditional_logits()`
  - Line 312-324: `_aa_convert()` alphabet conversion
  - Line 236-257: `_score()` internal scoring function

### PrxteinMPNN Key Files:
- `src/prxteinmpnn/model/mpnn.py` - Main model class
  - Line 130-187: `_call_unconditional()`
  - Line 189-250: `_call_conditional()`
  - Line 865-980: `__call__()` main entry point

- `src/prxteinmpnn/model/decoder.py` - Decoder logic
  - Line 237-258: Unconditional context preparation

- `src/prxteinmpnn/io/parsing/mappings.py` - Alphabet conversion
  - Line 34-43: `af_to_mpnn()` and `mpnn_to_af()`
  - Line 129-140: `residue_names_to_aatype()`

### Alphabet Reference:
```python
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET   = "ARNDCQEGHILKMFPSTWYVX"

# Position mapping:
# A: MPNN[0] = AF[0]
# R: MPNN[13] = AF[1]
# N: MPNN[11] = AF[2]
# D: MPNN[2] = AF[3]
# C: MPNN[1] = AF[4]
# etc.
```

---

## 🚀 Environment Setup

Before starting debugging, you need to set up the development environment properly.

### Clone ColabDesign Reference Implementation

```bash
# Clone ColabDesign to compare implementation details
cd /tmp
git clone https://github.com/sokrypton/ColabDesign.git

# Install ColabDesign (optional, for running comparison tests)
cd ColabDesign
pip install -e .
```

### Checkout and Update PrxteinMPNN Branch

```bash
# Navigate to PrxteinMPNN repository
cd /home/user/PrxteinMPNN

# Ensure you're on the validation branch
git checkout claude/prxteinmpnn-validation-cleanup-011CUuBUPG2G7Z9ArK9tw5Cv

# Pull latest changes (includes validation framework and this document)
git pull origin claude/prxteinmpnn-validation-cleanup-011CUuBUPG2G7Z9ArK9tw5Cv

# Install PrxteinMPNN in editable mode
uv sync

# Verify installation
python -c "import prxteinmpnn; print('PrxteinMPNN installed successfully')"
```

### Optional: Set Up Debug Branch

If you want to work on a separate debug branch (recommended for experimental changes):

```bash
# Create a new debug branch from the validation branch
git checkout -b debug/sequence-parsing-fix

# After making changes, commit and push
git add .
git commit -m "fix: correct sequence parsing alphabet conversion"
git push -u origin debug/sequence-parsing-fix
```

### Verify Environment

```bash
# Run a quick test to verify everything is working
cd /home/user/PrxteinMPNN
python -m pytest tests/validation/test_colabdesign_comparison.py -v

# Check that model weights are available
ls -lh src/prxteinmpnn/io/weights/
# Should see: original_v_48_020.eqx
```

### Quick Start Debugging Script

Create a minimal debugging script to start investigating:

```python
# debug_sequence_parsing.py
import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from colabdesign.mpnn.model import mk_mpnn_model

# Download test PDB
import os
if not os.path.exists("1ubq.pdb"):
    os.system("wget -q https://files.rcsb.org/download/1UBQ.pdb -O 1ubq.pdb")

print("="*80)
print("STEP 1: Parse with PrxteinMPNN")
print("="*80)
protein_tuple = next(parse_input("1ubq.pdb"))
protein = Protein.from_tuple(protein_tuple)
print(f"Sequence length: {protein.mask.sum():.0f}")
print(f"First 20 residues: {protein.aatype[:20]}")
print(f"aatype dtype: {protein.aatype.dtype}")
print(f"aatype shape: {protein.aatype.shape}")

print("\n" + "="*80)
print("STEP 2: Parse with ColabDesign")
print("="*80)
colab_model = mk_mpnn_model(model_name="v_48_020", weights="original")
colab_model.prep_inputs(pdb_filename="1ubq.pdb")
print(f"Sequence length: {colab_model._inputs['mask'].sum():.0f}")
print(f"First 20 residues: {colab_model._inputs['S'][:20]}")
print(f"S dtype: {colab_model._inputs['S'].dtype}")
print(f"S shape: {colab_model._inputs['S'].shape}")

print("\n" + "="*80)
print("STEP 3: Compare Sequences")
print("="*80)
if np.array_equal(protein.aatype, colab_model._inputs['S']):
    print("✅ Sequences match!")
else:
    print("❌ Sequences differ!")

    # Find differences
    diffs = np.where(protein.aatype != colab_model._inputs['S'])[0]
    print(f"Number of differences: {len(diffs)}")
    print(f"First 10 differences:")
    for i in diffs[:10]:
        print(f"  Position {i}: PrxteinMPNN={protein.aatype[i]}, ColabDesign={colab_model._inputs['S'][i]}")

print("\n" + "="*80)
print("STEP 4: Check Alphabet Conversion")
print("="*80)
from prxteinmpnn.io.parsing.mappings import MPNN_ALPHABET, AF_ALPHABET
print(f"MPNN alphabet: {MPNN_ALPHABET}")
print(f"AF alphabet:   {AF_ALPHABET}")
print("\nExample conversions:")
print(f"'A' in MPNN: index {MPNN_ALPHABET.index('A')}")
print(f"'A' in AF:   index {AF_ALPHABET.index('A')}")
print(f"'R' in MPNN: index {MPNN_ALPHABET.index('R')}")
print(f"'R' in AF:   index {AF_ALPHABET.index('R')}")
```

Run this script with:
```bash
cd /home/user/PrxteinMPNN
python debug_sequence_parsing.py
```

---

## ✅ Next Agent Instructions

### Context:
You are continuing validation work on the PrxteinMPNN implementation. Previous work revealed critical issues preventing the implementation from matching the ColabDesign reference. Your task is to debug these issues systematically.

### Prerequisites:
- Complete the "Environment Setup" section above
- Ensure both PrxteinMPNN and ColabDesign are installed
- Have model weights available in `src/prxteinmpnn/io/weights/`

### Immediate Tasks:

1. **Priority 1**: Fix sequence parsing mismatch
   - Run 1UBQ.pdb through both parsers
   - Print intermediate values at each conversion step
   - Ensure final integer sequences match exactly

2. **Priority 2**: Verify unconditional decoding
   - Add logging to trace encoder→decoder→logits flow
   - Compare intermediate values with ColabDesign
   - Check decoder context construction

3. **Priority 3**: Create minimal reproduction
   - Write a simple script that:
     - Loads 1UBQ
     - Runs unconditional scoring
     - Prints all intermediate values
     - Compares with ColabDesign at each step

### Success Criteria:

- [ ] Both implementations parse 1UBQ to identical sequences
- [ ] Encoder outputs match (or have explainable differences)
- [ ] Decoder contexts are constructed correctly
- [ ] Logits show similar patterns (high correlation, cosine similarity)
- [ ] Sequence recovery reaches 35-65% range

### Resources Available:

- Complete validation test suite in `tests/validation/`
- Colab notebook for interactive testing
- ColabDesign source code cloned to `/tmp/ColabDesign`
- All debugging tools and strategies outlined above

### Important Notes:

- Start with **sequence parsing** - if this is wrong, everything fails
- Use **small test cases** for faster iteration
- **Log everything** - we need to see where outputs diverge
- **Compare numerically** at each step using the provided comparison functions
- **Test incrementally** - fix one issue before moving to the next

Good luck! The codebase is well-structured, and the issue is likely localized to one or two specific points.

---

## 📞 Contact & Escalation

If you discover:
- **Architectural mismatches**: Document thoroughly and consider reimplementation
- **Numerical instability**: Check JAX compilation, dtype issues
- **Fundamental algorithm differences**: May need to consult ProteinMPNN paper

Remember: The goal is **exact reproduction** of ColabDesign behavior, which itself should match the original ProteinMPNN paper's reported performance.

---

## 🎯 Prompt for Next Agent

Use this prompt to start the next debugging session:

```
I need you to debug critical issues in the PrxteinMPNN implementation that are preventing it from matching the ColabDesign reference implementation.

CONTEXT:
- Previous validation revealed that PrxteinMPNN produces very different logits and poor sequence recovery (5.7% vs expected 35-65%)
- The main issues appear to be:
  1. Sequence parsing produces different integer arrays from the same PDB file
  2. Unconditional logits are low-confidence (narrow range) instead of high-confidence (full range)
  3. Overall performance is far below expected benchmarks

SETUP:
1. Read DEBUGGING_HANDOFF.md for complete context and analysis
2. Follow the "Environment Setup" section to clone ColabDesign and set up the branch
3. Run the "Quick Start Debugging Script" to verify the sequence parsing issue

TASK:
Start with Priority 1 (highest impact): Fix the sequence parsing mismatch
- Both implementations should parse 1UBQ.pdb to identical integer sequences (in MPNN alphabet)
- Use the debug script in DEBUGGING_HANDOFF.md as a starting point
- Add logging to trace the conversion at each step
- Compare with ColabDesign's parsing to identify where they diverge

Once sequence parsing is verified, move to Priority 2: Verify unconditional decoding path
- Compare encoder outputs between implementations
- Check decoder context construction
- Trace logits computation

SUCCESS CRITERIA:
- PrxteinMPNN and ColabDesign produce identical sequences from same PDB
- Logits show high correlation (Pearson > 0.9, Cosine similarity > 0.9)
- Sequence recovery reaches 35-65% range

Read DEBUGGING_HANDOFF.md first for the complete debugging strategy, code examples, and file references.
```

Copy and paste this prompt when starting the next session.
