"""
Investigate if W_out weights have alphabet reordering.

The hypothesis: ColabDesign's W_out might be in MPNN alphabet order internally,
but when combined with the alphabet conversion in the wrapper, the final logits
end up in AF alphabet order.

OR: The W_out weights themselves might be reordered to match AF alphabet.
"""

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.io.weights import load_model as load_prxteinmpnn
from colabdesign.mpnn.model import mk_mpnn_model

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

print("="*80)
print("Loading Models")
print("="*80)

# Load models
prx_model = load_prxteinmpnn(local_path="src/prxteinmpnn/io/weights/original_v_48_020.eqx")
colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)

print("\n" + "="*80)
print("Extracting W_out weights")
print("="*80)

# PrxteinMPNN W_out
prx_w_out = np.array(prx_model.w_out.weight)  # shape: (21, 128)
prx_b_out = np.array(prx_model.w_out.bias)    # shape: (21,)

print(f"PrxteinMPNN W_out:")
print(f"  Weight shape: {prx_w_out.shape}")
print(f"  Bias shape: {prx_b_out.shape}")
print(f"  Weight range: [{prx_w_out.min():.4f}, {prx_w_out.max():.4f}]")
print(f"  Bias range: [{prx_b_out.min():.4f}, {prx_b_out.max():.4f}]")

# ColabDesign W_out
colab_params = colab_model._model.params
colab_w_out_dict = colab_params['protein_mpnn/~/W_out']
print(f"\nColabDesign W_out dict keys: {list(colab_w_out_dict.keys())}")

# Extract weight and bias
colab_w_out = np.array(colab_w_out_dict['w']).T  # Transpose to match PrxteinMPNN shape
colab_b_out = np.array(colab_w_out_dict['b'])

print(f"ColabDesign W_out:")
print(f"  Weight shape (after transpose): {colab_w_out.shape}")
print(f"  Bias shape: {colab_b_out.shape}")
print(f"  Weight range: [{colab_w_out.min():.4f}, {colab_w_out.max():.4f}]")
print(f"  Bias range: [{colab_b_out.min():.4f}, {colab_b_out.max():.4f}]")

print("\n" + "="*80)
print("Direct Comparison (assuming same order)")
print("="*80)

# Direct comparison
diff_w = np.abs(prx_w_out - colab_w_out)
diff_b = np.abs(prx_b_out - colab_b_out)

print(f"Weight difference:")
print(f"  Max: {diff_w.max():.6f}")
print(f"  Mean: {diff_w.mean():.6f}")
print(f"  Correlation: {np.corrcoef(prx_w_out.flatten(), colab_w_out.flatten())[0, 1]:.6f}")

print(f"\nBias difference:")
print(f"  Max: {diff_b.max():.6f}")
print(f"  Mean: {diff_b.mean():.6f}")
print(f"  Correlation: {np.corrcoef(prx_b_out, colab_b_out)[0, 1]:.6f}")

if diff_w.max() < 1e-5:
    print("\n✅ Weights match directly - no reordering!")
else:
    print("\n❌ Weights don't match directly - checking with alphabet reordering...")

    print("\n" + "="*80)
    print("Testing Alphabet Reordering Hypothesis")
    print("="*80)

    # Hypothesis: ColabDesign W_out rows might be in AF alphabet order
    # Create permutation to reorder from AF to MPNN
    af_to_mpnn_perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])

    # Reorder ColabDesign weights to MPNN order
    colab_w_out_reordered = colab_w_out[af_to_mpnn_perm, :]
    colab_b_out_reordered = colab_b_out[af_to_mpnn_perm]

    diff_w_reordered = np.abs(prx_w_out - colab_w_out_reordered)
    diff_b_reordered = np.abs(prx_b_out - colab_b_out_reordered)

    print(f"After reordering ColabDesign from AF to MPNN order:")
    print(f"Weight difference:")
    print(f"  Max: {diff_w_reordered.max():.6f}")
    print(f"  Mean: {diff_w_reordered.mean():.6f}")
    print(f"  Correlation: {np.corrcoef(prx_w_out.flatten(), colab_w_out_reordered.flatten())[0, 1]:.6f}")

    print(f"\nBias difference:")
    print(f"  Max: {diff_b_reordered.max():.6f}")
    print(f"  Mean: {diff_b_reordered.mean():.6f}")
    print(f"  Correlation: {np.corrcoef(prx_b_out, colab_b_out_reordered)[0, 1]:.6f}")

    if diff_w_reordered.max() < 1e-5:
        print("\n✅ Weights match after AF→MPNN reordering!")
        print("   ColabDesign W_out is in AF alphabet order internally.")
    else:
        # Try the reverse - maybe PrxteinMPNN is in AF order?
        mpnn_to_af_perm = np.array([MPNN_ALPHABET.index(aa) for aa in AF_ALPHABET])
        prx_w_out_reordered = prx_w_out[mpnn_to_af_perm, :]
        prx_b_out_reordered = prx_b_out[mpnn_to_af_perm]

        diff_w_reverse = np.abs(prx_w_out_reordered - colab_w_out)
        diff_b_reverse = np.abs(prx_b_out_reordered - colab_b_out)

        print(f"\nTrying reverse - reordering PrxteinMPNN from MPNN to AF order:")
        print(f"Weight difference:")
        print(f"  Max: {diff_w_reverse.max():.6f}")
        print(f"  Mean: {diff_w_reverse.mean():.6f}")

        if diff_w_reverse.max() < 1e-5:
            print("\n✅ Weights match after MPNN→AF reordering of PrxteinMPNN!")
            print("   PrxteinMPNN W_out might be in AF alphabet order internally.")
        else:
            print("\n❌ Weights don't match with alphabet reordering either!")
            print("   This suggests fundamental weight differences.")

print("\n" + "="*80)
print("Examining Weight Patterns")
print("="*80)

# Look at first few rows of each
print("\nFirst 5 amino acids in each alphabet:")
print(f"MPNN: {MPNN_ALPHABET[:5]}")
print(f"AF:   {AF_ALPHABET[:5]}")

print("\nPrxteinMPNN W_out bias (first 5):")
for i in range(5):
    aa = MPNN_ALPHABET[i]
    print(f"  Row {i} ({aa}): {prx_b_out[i]:.6f}")

print("\nColabDesign W_out bias (first 5):")
for i in range(5):
    aa_in_af = AF_ALPHABET[i] if i < len(AF_ALPHABET) else '?'
    aa_in_mpnn = MPNN_ALPHABET[i]
    print(f"  Row {i} (AF:{aa_in_af}, MPNN:{aa_in_mpnn}): {colab_b_out[i]:.6f}")

# Check if any row in ColabDesign matches PrxteinMPNN row 0
print("\nSearching for PrxteinMPNN row 0 in ColabDesign...")
prx_row0 = prx_b_out[0]  # Should be 'A' in MPNN alphabet
print(f"PrxteinMPNN row 0 ('A'): {prx_row0:.6f}")

for i in range(21):
    if abs(colab_b_out[i] - prx_row0) < 1e-4:
        aa_at_i = AF_ALPHABET[i] if i < len(AF_ALPHABET) else '?'
        print(f"  Found match at ColabDesign row {i} ({aa_at_i}): {colab_b_out[i]:.6f}")

print("\n" + "="*80)
print("Summary")
print("="*80)
print("""
This analysis helps determine:
1. Whether weights are identical (just reordered by alphabet)
2. Or if weights are fundamentally different
3. Which alphabet order each implementation uses internally
""")
