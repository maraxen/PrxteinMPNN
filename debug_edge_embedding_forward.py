"""Debug the forward pass of edge_embedding to find the divergence."""

import jax
import jax.numpy as jnp
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from colabdesign.mpnn import mk_mpnn_model

print("="*80)
print("EDGE_EMBEDDING FORWARD PASS DEBUG")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"

# 1. Get the concatenated edges from both implementations
print("\n1. Running ColabDesign to get edges...")
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)

# We need to manually run through features to capture edges before embedding
inputs = mpnn_model._inputs
X = inputs['X']
Y = X.swapaxes(0, 1)
if Y.shape[0] == 4:
    b, c = (Y[1]-Y[0]), (Y[2]-Y[1])
    Cb = -0.58273431*jnp.cross(b,c) + 0.56802827*b - 0.54067466*c + Y[1]
    Y = jnp.concatenate([Y, Cb[None]], 0)

# Get neighbor indices
mask = inputs['mask']
def get_edge_idx_colabdesign(X_ca, mask, k=48):
    mask_2D = mask[...,None,:] * mask[...,:,None]
    dX = X_ca[...,None,:,:] - X_ca[...,:,None,:]
    D = jnp.sqrt(jnp.square(dX).sum(-1) + 1e-6)
    D_masked = jnp.where(mask_2D, D, D.max(-1, keepdims=True))
    return jax.lax.approx_min_k(D_masked, k, reduction_dimension=-1)[1]

E_idx = get_edge_idx_colabdesign(Y[1], mask, k=48)

# Compute RBF and positional features (simplified - we'll just use test case)
print(f"   Y shape: {Y.shape}")
print(f"   E_idx shape: {E_idx.shape}")

# 2. Load both models
print("\n2. Loading models...")
key = jax.random.PRNGKey(42)
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
params = joblib.load(colab_weights_path)['model_state_dict']
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

# 3. Test with a single edge vector
print("\n3. Testing with actual edge vector from position [0, 0]...")
print("   (We'll manually extract this from the real computation)")

# Load protein for PrxteinMPNN
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)

# Get PrxteinMPNN edge features (before w_e)
prx_edge_features, prx_neighbor_indices, _ = prx_model.features(
    key,
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    None,
)

# The output we got is AFTER norm_edges, so this doesn't help
# We need to capture the intermediate values

print("\n4. Check if equinox Linear and haiku Linear behave the same...")
# Create a test input
test_edge = jnp.array([-0.09738271, 0.06643821, -0.08947827, -0.2426403, 0.04809081] + [0.0] * 411)
print(f"   Test input shape: {test_edge.shape}")
print(f"   Test input[:5]: {test_edge[:5]}")

# ColabDesign forward (manual)
colab_w = params['protein_mpnn/~/protein_features/~/edge_embedding']['w']
colab_out = test_edge @ colab_w
print(f"\n   ColabDesign output (test_edge @ w):")
print(f"     Shape: {colab_out.shape}")
print(f"     [:5]: {colab_out[:5]}")

# PrxteinMPNN forward
prx_out = prx_model.features.w_e(test_edge)
print(f"\n   PrxteinMPNN output (w_e(test_edge)):")
print(f"     Shape: {prx_out.shape}")
print(f"     [:5]: {prx_out[:5]}")

match = jnp.allclose(colab_out, prx_out, atol=1e-5)
print(f"\n   Outputs match: {match}")
if not match:
    print(f"   Max diff: {jnp.max(jnp.abs(colab_out - prx_out))}")
