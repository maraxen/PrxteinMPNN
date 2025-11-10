"""Detailed test to verify architecture is correct."""

import jax
import jax.numpy as jnp
import numpy as np
from colabdesign.mpnn import mk_mpnn_model
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

print("="*80)
print("DETAILED ARCHITECTURE VERIFICATION")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"

# 1. ColabDesign
print("\n1. ColabDesign...")
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
colab_logits = mpnn_model.get_unconditional_logits()
print(f"   Logits shape: {colab_logits.shape}")
print(f"   First residue logits: {colab_logits[0,:5]}")

# 2. PrxteinMPNN
print("\n2. PrxteinMPNN...")
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

print(f"   Protein coordinates shape: {protein.coordinates.shape}")
print(f"   First residue N: {protein.coordinates[0, 0]}")
print(f"   First residue CA: {protein.coordinates[0, 1]}")
print(f"   First residue C: {protein.coordinates[0, 2]}")
print(f"   First residue CB (index 3): {protein.coordinates[0, 3]}")
print(f"   First residue O (index 4): {protein.coordinates[0, 4]}")

key = jax.random.PRNGKey(42)
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

# Check if w_e_proj exists
print(f"\n3. Model architecture check...")
print(f"   features.w_e_proj exists: {hasattr(prx_model.features, 'w_e_proj')}")
print(f"   model.w_e exists: {hasattr(prx_model, 'w_e')}")

_, prx_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",
    prng_key=key,
)

print(f"\n4. PrxteinMPNN results...")
print(f"   Logits shape: {prx_logits.shape}")
print(f"   First residue logits: {prx_logits[0,:5]}")

# 5. Compare
print(f"\n5. Comparison...")
corr = np.corrcoef(colab_logits.flatten(), prx_logits.flatten())[0, 1]
max_diff = np.max(np.abs(colab_logits - prx_logits))
print(f"   Correlation: {corr:.6f}")
print(f"   Max diff: {max_diff:.6f}")

if corr > 0.99:
    print("\n✅ SUCCESS!")
elif corr > 0.85:
    print(f"\n🟡 Good (was 0.871, now {corr:.3f})")
else:
    print(f"\n❌ Issue (correlation: {corr:.3f})")
