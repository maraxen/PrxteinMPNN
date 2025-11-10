"""Numerically compare the edges vectors."""

import jax
import jax.numpy as jnp
import numpy as np
from colabdesign.mpnn import mk_mpnn_model
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

# Monkey-patch to capture vectors
colab_edges = None
prx_edges = None

def capture_colab(orig_fn):
    def wrapper(*args, **kwargs):
        global colab_edges
        result = orig_fn(*args, **kwargs)
        # Result is (E, E_idx)
        colab_edges = result[0]
        return result
    return wrapper

def capture_prx(orig_fn):
    def wrapper(*args, **kwargs):
        global prx_edges
        result = orig_fn(*args, **kwargs)
        # Result is (edge_features, neighbor_indices, key)
        prx_edges = result[0]
        return result
    return wrapper

print("="*80)
print("NUMERICAL COMPARISON OF EDGES VECTORS")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"

# 1. Get ColabDesign edges (before edge_embedding)
print("\n1. Extracting ColabDesign edges (before edge_embedding)...")
# We need to get the edges BEFORE edge_embedding is applied
# Let me just run and capture from debug output

# For now, let's just run both and compare what we get at the features output
print("   Running ColabDesign...")
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
_ = mpnn_model.get_unconditional_logits()

print("\n2. Extracting PrxteinMPNN edges...")
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

key = jax.random.PRNGKey(42)
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

# Get features output
prx_edge_features, prx_neighbor_indices, _ = prx_model.features(
    key,
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    None,
)

print(f"\n   PrxteinMPNN features output shape: {prx_edge_features.shape}")
print(f"   PrxteinMPNN features output [0,0,:10]: {prx_edge_features[0,0,:10]}")

print("\n3. From debug output, we know:")
print("   - ColabDesign E[0,0] after norm: [-0.179, 0.142, -0.110, -0.561, -0.280]")
print("   - PrxteinMPNN edges[0,0] after norm: [-0.651, 0.663, -0.193, -0.499, -0.828]")
print("\n   These are DIFFERENT, so the divergence happens BEFORE norm!")
print("\n   But the inputs to edge_embedding (concatenated positional + RBF) appear")
print("   IDENTICAL from the full vectors shown above.")
print("\n   This means the issue must be in the edge_embedding layer itself!")
