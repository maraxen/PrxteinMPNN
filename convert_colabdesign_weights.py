"""Convert ColabDesign weights to PrxteinMPNN format."""

import joblib
import jax
import numpy as np
from prxteinmpnn.model import PrxteinMPNN
import equinox as eqx

# Load ColabDesign weights
print("Loading ColabDesign weights...")
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
checkpoint = joblib.load(colab_weights_path)
colab_params = checkpoint['model_state_dict']

print(f"Loaded {len(colab_params)} parameter groups")
print(f"k_neighbors: {checkpoint['num_edges']}")

# Create PrxteinMPNN skeleton
print("\nCreating PrxteinMPNN skeleton...")
key = jax.random.PRNGKey(0)
model = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=512,
    num_encoder_layers=3,
    num_decoder_layers=3,
    vocab_size=21,
    k_neighbors=checkpoint['num_edges'],
    key=key,
)

print("✅ Model skeleton created")
print(f"Model has {sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))} parameters")

# Save as a pickle for now - we can use this to understand the structure
print("\nSaving ColabDesign weights as numpy dict...")
np.savez("colabdesign_weights.npz", **{k: np.array(v) for k, v in colab_params.items()})
print("✅ Saved to colabdesign_weights.npz")

# Print weight names to understand mapping
print("\nColabDesign weight keys:")
def print_tree(d, prefix=""):
    for key in sorted(d.keys()):
        val = d[key]
        if isinstance(val, dict):
            print(f"{prefix}{key}:")
            print_tree(val, prefix + "  ")
        else:
            print(f"{prefix}{key:40s} {np.array(val).shape}")

print_tree(colab_params)
