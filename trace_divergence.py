"""
Layer-by-layer trace to find where PrxteinMPNN and ColabDesign diverge.
Weights are identical, so the issue is in the forward pass.
"""

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from colabdesign.mpnn.model import mk_mpnn_model
from colabdesign.mpnn.modules import RunModel
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

# Load test PDB
print("Loading test structure...")
pdb_path = "tests/data/1ubq.pdb"
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
print(f"Loaded {int(protein.mask.sum())} residues\n")

# Load models
print("Loading models...")
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
key = jax.random.PRNGKey(42)
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
colab_model.prep_inputs(pdb_filename=pdb_path)
print("✅ Models loaded\n")

print("="*80)
print("TRACING FORWARD PASS")
print("="*80)

# Step 1: Compare inputs
print("\n1. INPUT COORDINATES:")
prx_coords = protein.coordinates[:, :4, :]  # PrxteinMPNN uses first 4 atoms
colab_coords = colab_model._inputs['X']  # ColabDesign coords

print(f"  PrxteinMPNN coords shape: {prx_coords.shape}")
print(f"  ColabDesign coords shape: {colab_coords.shape}")
print(f"  Max coord difference: {np.abs(prx_coords - colab_coords).max():.6f}")
if np.abs(prx_coords - colab_coords).max() < 1e-5:
    print("  ✅ Coordinates match!")
else:
    print("  ❌ Coordinates differ!")

# Step 2: Feature extraction
print("\n2. FEATURE EXTRACTION:")
print("  Extracting features manually...")

# For PrxteinMPNN, we need to manually call the features module
prx_edge_features, prx_neighbor_indices, _ = prx_model.features(
    key,
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    None,  # No backbone noise for unconditional
)

print(f"  PrxteinMPNN edge_features shape: {prx_edge_features.shape}")
print(f"  PrxteinMPNN neighbor_indices shape: {prx_neighbor_indices.shape}")
print(f"  PrxteinMPNN edge_features range: [{prx_edge_features.min():.3f}, {prx_edge_features.max():.3f}]")

# For ColabDesign, we need to extract features from the internal model
# This is tricky because ColabDesign uses Haiku, so we need to call the model
# Let me just run the full model and see what we get

print("\n3. FULL FORWARD PASS:")

# Run PrxteinMPNN
_, prx_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",
    prng_key=key,
)

print(f"  PrxteinMPNN logits shape: {prx_logits.shape}")
print(f"  PrxteinMPNN logits range: [{prx_logits.min():.3f}, {prx_logits.max():.3f}]")
print(f"  PrxteinMPNN logits[0, :5]: {prx_logits[0, :5]}")

# Run ColabDesign
colab_logits_af = colab_model.get_unconditional_logits(key=key)
print(f"  ColabDesign logits (AF order) shape: {colab_logits_af.shape}")
print(f"  ColabDesign logits (AF order) range: [{colab_logits_af.min():.3f}, {colab_logits_af.max():.3f}]")

# Convert to MPNN order
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
colab_logits_mpnn = np.array(colab_logits_af)[..., perm]
print(f"  ColabDesign logits (MPNN order)[0, :5]: {colab_logits_mpnn[0, :5]}")

print("\n4. COMPARISON:")
corr = np.corrcoef(prx_logits.flatten(), colab_logits_mpnn.flatten())[0, 1]
print(f"  Correlation: {corr:.4f}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("""
The weights are identical (verified with correlation = 1.0).
But the logits correlation is only 0.06.

This means there's a fundamental difference in the forward pass computation.

Possible causes:
1. Different backbone noise application (even though we set it to 0?)
2. Different normalization epsilon values
3. Different scaling factors (e.g., scale=30.0 in encoder/decoder)
4. Different feature extraction (RBF, positional encoding)
5. Missing operations or incorrect order

Next step: Instrument the source code to print intermediate values.
""")

print("\nLet me check if there's augmentation noise...")
print(f"PrxteinMPNN augment_eps: {prx_model.features}")
