"""
Check specific model parameters that might differ.
"""

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.io.weights import load_model as load_prxteinmpnn
from colabdesign.mpnn.model import mk_mpnn_model
import joblib

print("="*80)
print("Checking Model Configuration Parameters")
print("="*80)

# Load PrxteinMPNN
prx_model = load_prxteinmpnn(local_path="src/prxteinmpnn/io/weights/original_v_48_020.eqx")

# Load ColabDesign
colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)

# Check ColabDesign config
print("\nColabDesign config:")
config = colab_model._model.config
for key, val in config.items():
    print(f"  {key}: {val}")

# Check if PrxteinMPNN has equivalent params
print("\nPrxteinMPNN parameters:")
print(f"  node_features_dim: {prx_model.node_features_dim}")
print(f"  edge_features_dim: {prx_model.edge_features_dim}")

# Check k_neighbors specifically
print(f"\n{'Parameter':<30} {'PrxteinMPNN':<20} {'ColabDesign':<20} {'Match?'}")
print("-" * 75)

# Try to find k_neighbors in PrxteinMPNN
if hasattr(prx_model, 'k_neighbors'):
    prx_k = prx_model.k_neighbors
elif hasattr(prx_model.features, 'k_neighbors'):
    prx_k = prx_model.features.k_neighbors
else:
    # Check in features module
    prx_k = "Unknown"

colab_k = config['k_neighbors']

print(f"{'k_neighbors':<30} {str(prx_k):<20} {str(colab_k):<20} {' ✅' if prx_k == colab_k else '❌'}")

# Check checkpoint to see what k_neighbors should be
print("\n" + "="*80)
print("Checking Original Checkpoint")
print("="*80)

checkpoint_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
try:
    checkpoint = joblib.load(checkpoint_path)
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print(f"  num_edges (k_neighbors): {checkpoint.get('num_edges', 'NOT FOUND')}")

    # Check if there are any other relevant params
    for key in checkpoint.keys():
        if key not in ['model_state_dict']:
            print(f"  {key}: {checkpoint[key]}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")

# Check bias usage
print("\n" + "="*80)
print("Checking Bias Parameter")
print("="*80)

print("\nColabDesign inputs after prep_inputs:")
colab_model.prep_inputs(pdb_filename="tests/data/1ubq.pdb")
print(f"  Has 'bias' key: {'bias' in colab_model._inputs}")
if 'bias' in colab_model._inputs:
    bias = colab_model._inputs['bias']
    print(f"  Bias shape: {bias.shape}")
    print(f"  Bias nonzero elements: {np.count_nonzero(bias)}")
    print(f"  Bias range: [{bias.min():.6f}, {bias.max():.6f}]")

print("\nPrxteinMPNN:")
print(f"  Has bias parameter: {hasattr(prx_model, 'bias') or hasattr(prx_model.w_out, 'bias')}")
if hasattr(prx_model.w_out, 'bias'):
    prx_bias = np.array(prx_model.w_out.bias)
    print(f"  w_out bias shape: {prx_bias.shape}")
    print(f"  w_out bias range: [{prx_bias.min():.6f}, {prx_bias.max():.6f}]")

# Check dropout
print("\n" + "="*80)
print("Checking Dropout and Augmentation")
print("="*80)

print(f"\nColabDesign:")
print(f"  dropout: {config['dropout']}")
print(f"  augment_eps (backbone_noise): {config['augment_eps']}")

print(f"\nPrxteinMPNN:")
print(f"  (Equinox models typically don't have dropout in inference mode)")

# Check model mode
print("\n" + "="*80)
print("Checking Model Inference State")
print("="*80)

print("\nLet's verify both models are in inference mode (not training)...")
print("  ColabDesign uses dropout=0.0 → inference mode")
print("  PrxteinMPNN is Equinox → typically inference mode by default")
