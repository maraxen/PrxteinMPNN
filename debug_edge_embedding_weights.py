"""Check if edge_embedding weights are loaded correctly."""

import jax
import jax.numpy as jnp
import joblib
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

print("="*80)
print("EDGE_EMBEDDING WEIGHT COMPARISON")
print("="*80)

# Load ColabDesign weights
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
params = joblib.load(colab_weights_path)['model_state_dict']

# Load PrxteinMPNN model
key = jax.random.PRNGKey(42)
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

print("\n1. ColabDesign edge_embedding weights...")
colab_w = params['protein_mpnn/~/protein_features/~/edge_embedding']['w']
print(f"   Shape: {colab_w.shape}")
print(f"   First 3x3:\n{colab_w[:3, :3]}")

print("\n2. PrxteinMPNN features.w_e weights...")
prx_w = prx_model.features.w_e.weight
print(f"   Shape: {prx_w.shape}")
print(f"   First 3x3:\n{prx_w[:3, :3]}")

print("\n3. Check if they match (considering transpose)...")
# We transpose when loading, so PrxteinMPNN weight should equal ColabDesign weight.T
match_transpose = jnp.allclose(prx_w, colab_w.T, atol=1e-5)
print(f"   prx_w == colab_w.T: {match_transpose}")

if not match_transpose:
    print(f"   Max diff: {jnp.max(jnp.abs(prx_w - colab_w.T))}")
    print(f"\n   ⚠️  Weights don't match!")

    # Check if they're the same without transpose
    match_no_transpose = jnp.allclose(prx_w, colab_w, atol=1e-5)
    print(f"   prx_w == colab_w (no transpose): {match_no_transpose}")

print("\n4. Test forward pass with same input...")
test_input = jnp.ones((416,))

# Manual ColabDesign forward (no bias for edge_embedding)
colab_output = test_input @ colab_w
print(f"   ColabDesign output (input @ w): {colab_output[:5]}")

# PrxteinMPNN forward
prx_output = prx_model.features.w_e(test_input)
print(f"   PrxteinMPNN output: {prx_output[:5]}")

match_forward = jnp.allclose(colab_output, prx_output, atol=1e-5)
print(f"   Outputs match: {match_forward}")

if not match_forward:
    print(f"   Max diff: {jnp.max(jnp.abs(colab_output - prx_output))}")
