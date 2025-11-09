"""Check what the default final_activation does."""

import jax
import jax.numpy as jnp
import equinox as eqx
from functools import partial

_gelu = partial(jax.nn.gelu, approximate=False)

# Create MLP
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
print("FINAL ACTIVATION INVESTIGATION")
print("="*80)

# Test if final_activation is identity
test_input = jnp.ones(128)

print(f"\nFinal activation function: {mlp.final_activation}")
print(f"Type: {type(mlp.final_activation)}")

# Apply it
output = mlp.final_activation(test_input)

print(f"\nInput:  {test_input[:5]}")
print(f"Output: {output[:5]}")

is_identity = jnp.allclose(test_input, output)
print(f"\nIs identity (input == output): {is_identity}")

if not is_identity:
    print("\n⚠️  Final activation is NOT identity!")
    print("   This means Equinox MLP applies a transformation after the final layer")
    print("   that ColabDesign doesn't!")

# Check if it's just reshaping or actually transforming values
max_diff = jnp.max(jnp.abs(test_input - output))
mean_diff = jnp.mean(jnp.abs(test_input - output))
print(f"\nMax difference: {max_diff}")
print(f"Mean difference: {mean_diff}")

# Now test with MLP creation specifying final_activation=lambda x: x
mlp_explicit_identity = eqx.nn.MLP(
    in_size=512,
    out_size=128,
    width_size=128,
    depth=2,
    activation=_gelu,
    final_activation=lambda x: x,
    key=key,
)

print("\n" + "="*80)
print("MLP WITH EXPLICIT final_activation=lambda x: x")
print("="*80)

test_input_mlp = jnp.ones(512)
output_default = mlp(test_input_mlp)
output_explicit = mlp_explicit_identity(test_input_mlp)

print(f"Default MLP output[:5]: {output_default[:5]}")
print(f"Explicit identity MLP output[:5]: {output_explicit[:5]}")

match = jnp.allclose(output_default, output_explicit)
print(f"\nOutputs match: {match}")

if not match:
    print(f"Max diff: {jnp.max(jnp.abs(output_default - output_explicit))}")
    print("\n⚠️  Default final_activation is different from identity!")
else:
    print("\n✅ Default final_activation IS identity")
