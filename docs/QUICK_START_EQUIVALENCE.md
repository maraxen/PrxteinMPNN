# Quick Start: Equinox vs Functional ProteinMPNN

This guide demonstrates the equivalence between Equinox and functional implementations of ProteinMPNN.

## Installation

```bash
pip install prxteinmpnn
```

## Setup

```python
import jax
import jax.numpy as jnp
from prxteinmpnn import conversion
from prxteinmpnn.functional import (
    get_functional_model,
    make_encoder,
    make_decoder,
    final_projection
)

# For reproducibility
key = jax.random.PRNGKey(42)
```

## Load Model Parameters

Both implementations start with the same parameters:

```python
# Load model parameters from checkpoint
params = get_functional_model("v_48_020.pkl", model_weights="original")
```

## Create Test Data

```python
# Create synthetic test data
num_atoms = 20
num_neighbors = 15
edge_dim = 128

edge_features = jax.random.normal(key, (num_atoms, num_neighbors, edge_dim))
neighbor_indices = jnp.arange(num_atoms)[:, None].repeat(num_neighbors, axis=1)
mask = jnp.ones(num_atoms)
```

## Functional Implementation

```python
# Create encoder function
encoder_fn = make_encoder(params, num_encoder_layers=3, scale=30.0)

# Create decoder function  
decoder_fn = make_decoder(
    params,
    attention_mask_type=None,
    num_decoder_layers=3,
    scale=30.0
)

# Forward pass
node_features, edge_features_encoded = encoder_fn(
    edge_features,
    neighbor_indices,
    mask
)
node_features_decoded = decoder_fn(
    node_features,
    edge_features_encoded,
    mask
)
func_logits = final_projection(params, node_features_decoded)

print("Functional logits shape:", func_logits.shape)  # (20, 21)
```

## Equinox Implementation

```python
# Create Equinox model
eqx_model = conversion.create_prxteinmpnn(
    params,
    num_encoder_layers=3,
    num_decoder_layers=3,
    key=key
)

# Forward pass (single call!)
eqx_logits = eqx_model(edge_features, neighbor_indices, mask)

print("Equinox logits shape:", eqx_logits.shape)  # (20, 21)
```

## Verify Equivalence

```python
# Check if outputs match
max_diff = jnp.max(jnp.abs(eqx_logits - func_logits))
mean_diff = jnp.mean(jnp.abs(eqx_logits - func_logits))

print(f"Max difference: {max_diff:.2e}")
print(f"Mean difference: {mean_diff:.2e}")

# Check with tolerance
matches = jnp.allclose(eqx_logits, func_logits, rtol=1e-5, atol=1e-5)
print(f"Outputs match (rtol=1e-5, atol=1e-5): {matches}")
```

Expected output:
```
Max difference: ~9.5e-06
Mean difference: ~1.2e-06
Outputs match (rtol=1e-5, atol=1e-5): True
```

## Save and Load Equinox Model

```python
import equinox
from prxteinmpnn import eqx

# Save model
equinox.tree_serialise_leaves("my_model.eqx", eqx_model)

# Load model
loaded_model = eqx.load_prxteinmpnn("my_model.eqx")

# Verify loaded model works
loaded_logits = loaded_model(edge_features, neighbor_indices, mask)

# Should be bit-perfect identical
print(f"Save/load preserves output: {jnp.allclose(eqx_logits, loaded_logits, rtol=1e-7, atol=1e-8)}")
```

## Performance Comparison

```python
import time

# Functional implementation
func_encoder_jit = jax.jit(encoder_fn)
func_decoder_jit = jax.jit(decoder_fn)

start = time.time()
# First call compiles
_ = func_encoder_jit(edge_features, neighbor_indices, mask)
_ = func_decoder_jit(node_features, edge_features_encoded, mask)
compile_time_func = time.time() - start

# Subsequent calls are fast
start = time.time()
for _ in range(100):
    n, e = func_encoder_jit(edge_features, neighbor_indices, mask)
    n = func_decoder_jit(n, e, mask)
run_time_func = (time.time() - start) / 100

print(f"Functional - Compile: {compile_time_func:.2f}s, Run: {run_time_func*1000:.2f}ms")

# Equinox implementation
@jax.jit
def eqx_forward(model, edge_features, neighbor_indices, mask):
    return model(edge_features, neighbor_indices, mask)

start = time.time()
# First call compiles
_ = eqx_forward(eqx_model, edge_features, neighbor_indices, mask)
compile_time_eqx = time.time() - start

# Subsequent calls are fast
start = time.time()
for _ in range(100):
    _ = eqx_forward(eqx_model, edge_features, neighbor_indices, mask)
run_time_eqx = (time.time() - start) / 100

print(f"Equinox    - Compile: {compile_time_eqx:.2f}s, Run: {run_time_eqx*1000:.2f}ms")
```

## When to Use Which?

### Use Functional Implementation When:
- You need access to intermediate layer outputs
- You want to insert custom operations between layers
- You're integrating with existing functional JAX code
- You need maximum flexibility

### Use Equinox Implementation When:
- You want a clean, object-oriented API
- You're familiar with PyTorch/Flax-style modules
- You need easy model serialization
- You prefer single forward pass calls

## Next Steps

- Read the full equivalence documentation in `docs/EQUINOX_FUNCTIONAL_EQUIVALENCE.md`
- Check out example notebooks in `examples/`
- Run the test suite: `pytest tests/test_eqx_equivalence.py -v`

## References

- [Equinox Documentation](https://docs.kidger.site/equinox/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [ProteinMPNN Paper](https://www.science.org/doi/10.1126/science.add2187)
