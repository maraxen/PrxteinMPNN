"""Test if the fix improved correlation with real ColabDesign."""

import jax
import jax.numpy as jnp
import numpy as np
from colabdesign.mpnn import mk_mpnn_model
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

def compare(name, a, b):
    """Compare two arrays and return correlation."""
    a_flat = np.array(a).flatten()
    b_flat = np.array(b).flatten()
    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    max_diff = np.max(np.abs(a_flat - b_flat))
    mean_diff = np.mean(np.abs(a_flat - b_flat))
    print(f"{name}:")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    return corr

print("="*80)
print("TESTING FIX: PrxteinMPNN vs REAL ColabDesign")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"

# 1. Real ColabDesign
print("\n1. Running REAL ColabDesign...")
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
colab_logits = mpnn_model.get_unconditional_logits()
print(f"   Shape: {colab_logits.shape}")
print(f"   Sample [0,:5]: {colab_logits[0,:5]}")

# 2. PrxteinMPNN
print("\n2. Running PrxteinMPNN...")
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

key = jax.random.PRNGKey(42)
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

_, prx_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",  # decoding_approach
    prng_key=key,
)
print(f"   Shape: {prx_logits.shape}")
print(f"   Sample [0,:5]: {prx_logits[0,:5]}")

# 3. Compare
print("\n3. Comparison...")
print("="*80)
corr = compare("Final logits", colab_logits, prx_logits)

print("\n" + "="*80)
if corr > 0.99:
    print("✅ SUCCESS! Correlation > 0.99")
    print("   The fix worked!")
elif corr > 0.90:
    print("🟡 GOOD! Correlation > 0.90 but < 0.99")
    print(f"   Improved from 0.871 to {corr:.3f}")
elif corr > 0.871:
    print(f"🟡 BETTER! Correlation improved from 0.871 to {corr:.3f}")
else:
    print(f"❌ WORSE! Correlation is {corr:.3f} (was 0.871)")
print("="*80)
