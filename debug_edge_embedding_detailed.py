"""Debug edge_embedding application in detail."""

import jax
import jax.numpy as jnp
import joblib
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

print("="*80)
print("EDGE_EMBEDDING DETAILED DEBUG")
print("="*80)

# Load weights and model
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
params = joblib.load(colab_weights_path)['model_state_dict']
key = jax.random.PRNGKey(42)
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

print("\n1. Check edge_embedding weight shapes...")
colab_w = params['protein_mpnn/~/protein_features/~/edge_embedding']['w']
print(f"   ColabDesign weight shape: {colab_w.shape}")  # (416, 128)
print(f"   PrxteinMPNN weight shape: {prx_model.features.w_e.weight.shape}")  # Should be (128, 416)

print("\n2. Check if weights match (with transpose)...")
match = jnp.allclose(prx_model.features.w_e.weight, colab_w.T, atol=1e-5)
print(f"   prx.weight == colab.w.T: {match}")

print("\n3. Test with actual edge vector from debug output...")
# From debug output, the input edges are identical:
test_edge = jnp.array([-0.09738271, 0.06643821, -0.08947827, -0.2426403, 0.04809081] + [0.0] * 411)

# Manual ColabDesign forward (input @ w)
colab_out = test_edge @ colab_w
print(f"\n   ColabDesign forward (test_edge @ w):")
print(f"     Output[:5]: {colab_out[:5]}")
print(f"     Expected (from debug): [-0.4029066, 0.08976443, -0.10309178, -0.5805124, -0.6815348]")

# PrxteinMPNN forward
prx_out = prx_model.features.w_e(test_edge)
print(f"\n   PrxteinMPNN forward (w_e(test_edge)):")
print(f"     Output[:5]: {prx_out[:5]}")
print(f"     Expected (from debug): [-1.3347393, 1.0927724, -0.28576708, -0.51301026, -1.6843283]")

# Check if PrxteinMPNN output matches ColabDesign
match_outputs = jnp.allclose(colab_out, prx_out, atol=1e-3)
print(f"\n   Outputs match: {match_outputs}")
if not match_outputs:
    print(f"   Max diff: {jnp.max(jnp.abs(colab_out - prx_out))}")

print("\n4. Check if eqx.nn.Linear with use_bias=False works correctly...")
print(f"   PrxteinMPNN w_e has bias: {hasattr(prx_model.features.w_e, 'bias') and prx_model.features.w_e.bias is not None}")

print("\n5. Manual matrix multiplication...")
# PrxteinMPNN weight is (128, 416), input is (416,)
# So we need: weight @ input
manual_out = prx_model.features.w_e.weight @ test_edge
print(f"   Manual (weight @ input): {manual_out[:5]}")

# ColabDesign does: input @ weight where weight is (416, 128)
# Which is equivalent to: weight.T @ input
print(f"   ColabDesign equivalent (weight.T @ input): {colab_w.T @ test_edge[:5]}")
