"""
Compare model weights between PrxteinMPNN and ColabDesign.

This script:
1. Loads weights from both implementations
2. Compares key weight matrices
3. Reports any differences that might explain logits mismatch
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from prxteinmpnn.io.weights import load_model as load_prxteinmpnn
from colabdesign.mpnn.model import mk_mpnn_model

print("="*80)
print("Loading Models and Weights")
print("="*80)

# Load PrxteinMPNN
local_weights_path = "src/prxteinmpnn/io/weights/original_v_48_020.eqx"
prx_model = load_prxteinmpnn(local_path=local_weights_path)
print("✅ PrxteinMPNN loaded")

# Load ColabDesign
colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
print("✅ ColabDesign loaded")

print("\n" + "="*80)
print("Model Architecture Comparison")
print("="*80)

# PrxteinMPNN architecture
print("\nPrxteinMPNN:")
# Check what attributes exist
attrs = [a for a in dir(prx_model) if not a.startswith('_') and not callable(getattr(prx_model, a))]
print(f"  Available attributes: {attrs[:15]}")

# Try to print architecture params
try:
    print(f"  Node features dim: {prx_model.node_features_dim}")
    print(f"  Edge features dim: {prx_model.edge_features_dim}")
    print(f"  Hidden features dim: {prx_model.hidden_features_dim}")
except:
    pass

# ColabDesign architecture
print("\nColabDesign (_model.params structure):")
print(f"  Config: {colab_model._model.config}")

print("\n" + "="*80)
print("Weight Matrix Comparison")
print("="*80)

def compare_weights(prx_weights, colab_weights, name, transpose=False):
    """Compare two weight matrices."""
    prx_w = np.array(prx_weights)
    colab_w = np.array(colab_weights)

    if transpose:
        # ColabDesign might use transposed weights
        colab_w = colab_w.T

    print(f"\n{name}:")
    print(f"  PrxteinMPNN shape: {prx_w.shape}")
    print(f"  ColabDesign shape: {colab_w.shape}")

    if prx_w.shape != colab_w.shape:
        print(f"  ❌ Shape mismatch!")
        return

    # Compare statistics
    print(f"  PrxteinMPNN: min={prx_w.min():.6f}, max={prx_w.max():.6f}, mean={prx_w.mean():.6f}, std={prx_w.std():.6f}")
    print(f"  ColabDesign: min={colab_w.min():.6f}, max={colab_w.max():.6f}, mean={colab_w.mean():.6f}, std={colab_w.std():.6f}")

    # Compare directly
    diff = np.abs(prx_w - colab_w)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    if max_diff < 1e-5:
        print(f"  ✅ Weights match!")
    elif max_diff < 1e-3:
        print(f"  ⚠️  Small differences (numerical precision?)")
    else:
        print(f"  ❌ Significant differences!")

    # Correlation
    corr = np.corrcoef(prx_w.flatten(), colab_w.flatten())[0, 1]
    print(f"  Correlation: {corr:.6f}")

    return max_diff, corr

print("\nExploring weight structure...")

# Try to access weights from both models
print("\nPrxteinMPNN model structure:")
print(f"  Type: {type(prx_model)}")

# Get encoder weights
try:
    print("\nEncoder layer 0:")
    prx_enc0 = prx_model.encoder.layers[0]
    print(f"  Type: {type(prx_enc0)}")

    # Try to access specific weights
    if hasattr(prx_enc0, 'W_Q'):
        print(f"  W_Q shape: {prx_enc0.W_Q.shape}")
    if hasattr(prx_enc0, 'W_K'):
        print(f"  W_K shape: {prx_enc0.W_K.shape}")
    if hasattr(prx_enc0, 'W_V'):
        print(f"  W_V shape: {prx_enc0.W_V.shape}")

    # Print all attributes
    attrs = [a for a in dir(prx_enc0) if not a.startswith('_')]
    print(f"  Attributes: {attrs[:10]}...")  # First 10
except Exception as e:
    print(f"  Error accessing encoder: {e}")

print("\nColabDesign params structure:")
# Explore ColabDesign params
colab_params = colab_model._model.params
print(f"  Type: {type(colab_params)}")
print(f"  Keys: {list(colab_params.keys())[:10] if hasattr(colab_params, 'keys') else 'Not a dict'}")

# Try to find encoder weights
if isinstance(colab_params, dict):
    for key in list(colab_params.keys())[:20]:
        val = colab_params[key]
        if hasattr(val, 'shape'):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        elif isinstance(val, dict):
            print(f"  {key}: <dict with {len(val)} keys>")
        else:
            print(f"  {key}: {type(val)}")

print("\n" + "="*80)
print("Attempting Direct Weight Comparison")
print("="*80)

# Try to compare some weights
try:
    # Example: Compare w_out (final output projection)
    # This should be easier to find and compare

    print("\nLooking for output projection weights (w_out)...")

    if hasattr(prx_model, 'w_out'):
        prx_w_out = prx_model.w_out
        print(f"PrxteinMPNN w_out:")
        # Check if it's a linear layer or raw weights
        if hasattr(prx_w_out, 'weight'):
            print(f"  Weight shape: {prx_w_out.weight.shape}")
            print(f"  Has bias: {hasattr(prx_w_out, 'bias')}")
        else:
            print(f"  Type: {type(prx_w_out)}")
            print(f"  Attributes: {[a for a in dir(prx_w_out) if not a.startswith('_')][:10]}")

    # Look for ColabDesign w_out
    if 'W_out' in colab_params:
        colab_w_out = colab_params['W_out']
        print(f"ColabDesign W_out shape: {colab_w_out.shape}")
    elif 'w_out' in colab_params:
        colab_w_out = colab_params['w_out']
        print(f"ColabDesign w_out shape: {colab_w_out.shape}")
    else:
        print("Could not find w_out in ColabDesign params")
        # Print all keys that might be w_out
        matching_keys = [k for k in colab_params.keys() if 'out' in k.lower()]
        print(f"Keys containing 'out': {matching_keys}")

except Exception as e:
    print(f"Error comparing weights: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("Summary")
print("="*80)

print("""
To properly compare weights, we need to:
1. Understand the exact structure of both model's parameters
2. Map corresponding layers between implementations
3. Account for any transposes or reshapes
4. Check if alphabet reordering affects weight matrices

Next step: Create a detailed layer-by-layer forward pass comparison
to see where outputs diverge.
""")
