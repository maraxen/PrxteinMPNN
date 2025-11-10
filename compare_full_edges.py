"""Compare full 416-dimensional edges vectors between implementations."""

import jax
import jax.numpy as jnp
import numpy as np
from colabdesign.mpnn import mk_mpnn_model
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

print("="*80)
print("CAPTURING FULL EDGES VECTORS")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"

# This will print debug output with full vectors
print("\n1. Running ColabDesign...")
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
_ = mpnn_model.get_unconditional_logits()

print("\n2. Running PrxteinMPNN...")
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

key = jax.random.PRNGKey(42)
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

_, _ = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",
    prng_key=key,
)

print("\n" + "="*80)
print("Check the debug output above for:")
print("  - ColabDesign: E[0,0] (FULL edge vector)")
print("  - PrxteinMPNN: edges[0,0] FULL VECTOR")
print("="*80)
