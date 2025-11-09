"""Test if vmapped eqx.nn.Linear behaves identically to manual computation."""

import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.stats import pearsonr

def compare(name, arr1, arr2):
    a1, a2 = arr1.flatten(), arr2.flatten()
    corr = pearsonr(a1, a2)[0] if len(a1) > 1 else 0.0
    max_diff = jnp.max(jnp.abs(a1 - a2))
    mean_diff = jnp.mean(jnp.abs(a1 - a2))
    status = "✅" if corr > 0.9999 else ("🟡" if corr > 0.99 else "❌")
    print(f"{status} {name}:")
    print(f"   Corr: {corr:.6f}, Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
    return corr

print("="*80)
print("TESTING EQUINOX VMAP BEHAVIOR")
print("="*80)

# Create test data
key = jax.random.PRNGKey(42)
test_data = jax.random.normal(key, (76, 48, 128))  # (N, K, C)

# Create a Linear layer
key, subkey = jax.random.split(key)
linear = eqx.nn.Linear(128, 128, key=subkey)

print(f"\nLinear layer weight shape: {linear.weight.shape}")
print(f"Linear layer bias shape: {linear.bias.shape}")

# Test 1: Single application
print("\n1. Single application (no vmap)...")
single_input = test_data[0, 0, :]  # (128,)
output_single = linear(single_input)
output_manual = single_input @ linear.weight.T + linear.bias
compare("  Single: eqx.nn.Linear vs manual", output_single, output_manual)

# Test 2: Single vmap
print("\n2. Single vmap (over K dimension)...")
batch_input = test_data[0, :, :]  # (48, 128)
output_vmap1 = jax.vmap(linear)(batch_input)
output_manual1 = jax.vmap(lambda x: x @ linear.weight.T + linear.bias)(batch_input)
compare("  Single vmap: eqx.nn.Linear vs manual", output_vmap1, output_manual1)

# Test 3: Double vmap
print("\n3. Double vmap (over N and K dimensions)...")
output_vmap2 = jax.vmap(jax.vmap(linear))(test_data)
output_manual2 = jax.vmap(jax.vmap(lambda x: x @ linear.weight.T + linear.bias))(test_data)
compare("  Double vmap: eqx.nn.Linear vs manual", output_vmap2, output_manual2)

# Test 4: LayerNorm
print("\n4. LayerNorm...")
layer_norm = eqx.nn.LayerNorm(128)

single_output_ln = layer_norm(single_input)

# Manual LayerNorm
mean = single_input.mean()
var = single_input.var()
manual_ln = ((single_input - mean) / jnp.sqrt(var + 1e-5)) * layer_norm.weight + layer_norm.bias

compare("  Single LayerNorm: eqx vs manual", single_output_ln, manual_ln)

# Test 5: Double vmap LayerNorm
print("\n5. Double vmap LayerNorm...")
output_ln_vmap = jax.vmap(jax.vmap(layer_norm))(test_data)

def manual_ln_fn(x):
    mean = x.mean()
    var = x.var()
    return ((x - mean) / jnp.sqrt(var + 1e-5)) * layer_norm.weight + layer_norm.bias

output_ln_manual = jax.vmap(jax.vmap(manual_ln_fn))(test_data)

compare("  Double vmap LayerNorm: eqx vs manual", output_ln_vmap, output_ln_manual)

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If all tests show perfect correlation, Equinox layers are not the issue.")
print("If there's divergence, we found a bug in how Equinox layers work with vmap!")
