"""Compare real ColabDesign edge features with PrxteinMPNN using debug prints."""

import jax
import jax.numpy as jnp
from colabdesign.mpnn import mk_mpnn_model
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

print("="*80)
print("REAL COLABDESIGN WITH DEBUG PRINTS")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"

# 1. Run real ColabDesign (will print debug info)
print("\n1. Running REAL ColabDesign...")
print("-"*80)
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
colab_logits = mpnn_model.get_unconditional_logits()
print("-"*80)
print(f"   ColabDesign logits shape: {colab_logits.shape}")
print(f"   ColabDesign logits[0,:5]: {colab_logits[0,:5]}")

# 2. Run PrxteinMPNN features
print("\n2. Running PrxteinMPNN features...")
print("-"*80)

# Load protein
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

# Load PrxteinMPNN model
key = jax.random.PRNGKey(42)
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

# Run features
prx_edge_features, prx_neighbor_indices, _ = prx_model.features(
    key,
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    None,
)

print("-"*80)
print(f"   PrxteinMPNN edge features shape: {prx_edge_features.shape}")
print(f"   PrxteinMPNN neighbor indices shape: {prx_neighbor_indices.shape}")
print(f"   PrxteinMPNN edge_features[0,0,:5]: {prx_edge_features[0,0,:5]}")
print(f"   PrxteinMPNN neighbor_indices[0,:5]: {prx_neighbor_indices[0,:5]}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)
print("Look at the debug prints above to see:")
print("  - How ColabDesign computes C-beta")
print("  - What shapes each operation produces")
print("  - The exact values at each step")
print("\nThen compare with PrxteinMPNN's implementation to find divergence.")
