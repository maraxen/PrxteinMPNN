"""Test Equinox MLP structure to verify it matches ColabDesign."""

import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial


_gelu = partial(jax.nn.gelu, approximate=False)

# Create an MLP like in decoder
key = jax.random.PRNGKey(42)
mlp = eqx.nn.MLP(
    in_size=512,
    out_size=128,
    width_size=128,
    depth=2,
    activation=_gelu,
    key=key,
)

print("="*80)
print("EQUINOX MLP STRUCTURE")
print("="*80)

# Print the layers
print(f"\nMLP structure (depth=2):")
print(f"  Number of layers: {len(mlp.layers)}")

for i, layer in enumerate(mlp.layers):
    if hasattr(layer, 'weight'):
        print(f"  Layer {i}: Linear({layer.weight.shape[1]} -> {layer.weight.shape[0]})")
    else:
        print(f"  Layer {i}: {type(layer).__name__}")

print(f"\nActivation: {mlp.activation}")
print(f"Final activation: {mlp.final_activation}")

# Test forward pass structure
test_input = jnp.ones(512)

# Manual computation to verify
h = test_input
for i, layer in enumerate(mlp.layers[:-1]):
    h = layer(h)
    h = mlp.activation(h)
    print(f"\nAfter layer {i} + activation: shape {h.shape}")

# Final layer
h_final_manual = mlp.layers[-1](h)
print(f"After final layer (NO activation): shape {h_final_manual.shape}")

# Check if final_activation is applied
if mlp.final_activation:
    h_final_manual = mlp.activation(h_final_manual)
    print(f"After final activation: shape {h_final_manual.shape}")

# Compare with MLP forward
h_mlp = mlp(test_input)
print(f"\nMLP forward output: shape {h_mlp.shape}")

match = jnp.allclose(h_final_manual, h_mlp)
print(f"Manual matches MLP forward: {match}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"Depth=2 creates {len(mlp.layers)} layers")
print(f"Activation applied to: layers 0-{len(mlp.layers)-2}")
print(f"NO activation on final layer: {not mlp.final_activation}")

expected_structure = "Input -> Linear -> GELU -> Linear -> GELU -> Linear (no GELU)"
print(f"\nExpected structure:\n  {expected_structure}")
print("\nThis should match ColabDesign's W1->GELU->W2->GELU->W3 structure!")
