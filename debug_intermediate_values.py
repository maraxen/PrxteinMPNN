"""
Compare intermediate values with debug output enabled.
"""

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.io.weights import load_model as load_prxteinmpnn
from colabdesign.mpnn.model import mk_mpnn_model

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

def af_logits_to_mpnn(logits_af):
    """Convert logits from AF alphabet order to MPNN alphabet order."""
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    return logits_af[..., perm]

print("="*80)
print("SETUP")
print("="*80)

# Load test structure
pdb_path = "tests/data/1ubq.pdb"
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
print(f"✅ Loaded 1UBQ: {int(protein.mask.sum())} residues")

# Load models
prx_model = load_prxteinmpnn(local_path="src/prxteinmpnn/io/weights/original_v_48_020.eqx")
colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
colab_model.prep_inputs(pdb_filename=pdb_path)

key = jax.random.PRNGKey(42)

print("\n" + "="*80)
print("PRXTEINMPNN WITH DEBUG OUTPUT")
print("="*80)

# Run PrxteinMPNN with debug=True
_, prx_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",
    prng_key=key,
    debug=True,  # Enable debug output!
)

print("\n" + "="*80)
print("COLABDESIGN OUTPUT")
print("="*80)

colab_logits_af = colab_model.get_unconditional_logits(key=key)
colab_logits_mpnn = af_logits_to_mpnn(colab_logits_af)

print(f"\nColabDesign logits (after conversion to MPNN order):")
print(f"  Shape: {colab_logits_mpnn.shape}")
print(f"  Range: [{colab_logits_mpnn.min():.3f}, {colab_logits_mpnn.max():.3f}]")
print(f"  First position logits: {colab_logits_mpnn[0]}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# Compare logits
diff = np.abs(prx_logits - colab_logits_mpnn)
print(f"\nLogits difference:")
print(f"  Max: {diff.max():.6f}")
print(f"  Mean: {diff.mean():.6f}")
print(f"  Correlation: {np.corrcoef(prx_logits.flatten(), colab_logits_mpnn.flatten())[0, 1]:.6f}")

# Find position with largest difference
max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
print(f"\nLargest difference at position {max_diff_idx}:")
print(f"  PrxteinMPNN: {prx_logits[max_diff_idx]:.3f}")
print(f"  ColabDesign: {colab_logits_mpnn[max_diff_idx]:.3f}")
print(f"  Difference: {diff[max_diff_idx]:.3f}")

# Check if the ranges are systematically different
prx_max_per_pos = prx_logits.max(axis=1)
colab_max_per_pos = colab_logits_mpnn.max(axis=1)

print(f"\nMax logit per position comparison:")
print(f"  PrxteinMPNN max values: [{prx_max_per_pos.min():.3f}, {prx_max_per_pos.max():.3f}]")
print(f"  ColabDesign max values: [{colab_max_per_pos.min():.3f}, {colab_max_per_pos.max():.3f}]")
print(f"  Correlation of max values: {np.corrcoef(prx_max_per_pos, colab_max_per_pos)[0, 1]:.4f}")

# Check first position in detail
print(f"\nFirst position (index 0) logits:")
print(f"  PrxteinMPNN: {prx_logits[0][:10]}... (first 10)")
print(f"  ColabDesign: {colab_logits_mpnn[0][:10]}... (first 10)")
print(f"  Difference:  {(prx_logits[0] - colab_logits_mpnn[0])[:10]}... (first 10)")

# Check if there's a systematic scaling
ratio = colab_logits_mpnn / (prx_logits + 1e-8)  # Add epsilon to avoid division by zero
print(f"\nRatio ColabDesign/PrxteinMPNN:")
print(f"  Mean: {ratio.mean():.3f}")
print(f"  Std: {ratio.std():.3f}")
print(f"  This would suggest {'a systematic scaling factor' if ratio.std() < 0.5 else 'NOT a simple scaling'}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
The debug output from PrxteinMPNN shows intermediate values.
If encoder/decoder outputs match but final logits differ, the issue is in w_out.
If encoder outputs differ, the issue is in feature extraction or encoder.
If decoder outputs differ, the issue is in the decoder.

Compare these step by step with ColabDesign to find the divergence point.
""")
