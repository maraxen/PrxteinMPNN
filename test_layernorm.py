"""Compare LayerNorm computation."""

import jax
import jax.numpy as jnp
import equinox as eqx
import joblib
from scipy.stats import pearsonr

# Create test data
key = jax.random.PRNGKey(42)
test_data = jax.random.normal(key, (76, 48, 128))

# Load params
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
params = joblib.load(colab_weights_path)['model_state_dict']

scale = params['protein_mpnn/~/protein_features/~/norm_edges']['scale']
offset = params['protein_mpnn/~/protein_features/~/norm_edges']['offset']

print("="*80)
print("LAYERNORM COMPARISON")
print("="*80)

print("\n1. ColabDesign LayerNorm (axis=-1)...")
mean_colab = test_data.mean(axis=-1, keepdims=True)
var_colab = test_data.var(axis=-1, keepdims=True)
normed_colab = (test_data - mean_colab) / jnp.sqrt(var_colab + 1e-5)
normed_colab = normed_colab * scale + offset

print("\n2. PrxteinMPNN LayerNorm (vmap)...")
def norm_fn(x):
    mean = x.mean()
    var = x.var()
    return ((x - mean) / jnp.sqrt(var + 1e-5)) * scale + offset
normed_vmap = jax.vmap(jax.vmap(norm_fn))(test_data)

print("\n3. Equinox LayerNorm...")
layer_norm = eqx.nn.LayerNorm(128)
layer_norm = eqx.tree_at(lambda l: l.weight, layer_norm, scale)
layer_norm = eqx.tree_at(lambda l: l.bias, layer_norm, offset)
normed_eqx = jax.vmap(jax.vmap(layer_norm))(test_data)

print("\n" + "="*80)
print("COMPARISONS")
print("="*80)

def compare(name, arr1, arr2):
    a1, a2 = arr1.flatten(), arr2.flatten()
    corr = pearsonr(a1, a2)[0] if len(a1) > 1 else 0.0
    max_diff = jnp.max(jnp.abs(a1 - a2))
    mean_diff = jnp.mean(jnp.abs(a1 - a2))

    status = "✅" if corr > 0.9999 else ("🟡" if corr > 0.99 else "❌")
    print(f"{status} {name}:")
    print(f"   Corr: {corr:.6f}, Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

compare("ColabDesign vs vmap", normed_colab, normed_vmap)
compare("ColabDesign vs Equinox", normed_colab, normed_eqx)
compare("vmap vs Equinox", normed_vmap, normed_eqx)

print("\nConclusion:")
print("If all three match perfectly, LayerNorm is not the issue.")
print("If there's divergence, we found the root cause!")
