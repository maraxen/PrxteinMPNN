# Quick Start: New Equinox Implementation

This guide demonstrates the new Equinox implementation (`eqx_new.py`) and its equivalence with the functional implementation.

**Status**: ✅ All core equivalence tests passing (4/4, 11.6 seconds)

## Installation

```bash
pip install prxteinmpnn
```

## Setup

```python
import jax
import jax.numpy as jnp
from prxteinmpnn.eqx_new import PrxteinMPNN
from prxteinmpnn.functional import (
    get_functional_model,
    make_encoder,
    make_decoder,
    final_projection
)

# For reproducibility
key = jax.random.PRNGKey(42)
```

## Quick Example: New Equinox Model

```python
# Load functional parameters
params = get_functional_model("v_48_020", model_weights="original")

# Create new Equinox model
model = PrxteinMPNN.from_functional(
    params,
    num_encoder_layers=3,
    num_decoder_layers=3,
    key=key
)

# Create test data
num_residues = 25
K = 48  # neighbors
edge_features = jax.random.normal(key, (num_residues, K, 128))
neighbor_indices = jnp.tile(jnp.arange(num_residues)[:, None], (1, K))
mask = jnp.ones(num_residues)

# Simple forward pass - unconditional decoding
_, logits = model._call_unconditional(edge_features, neighbor_indices, mask)
print(f"Logits shape: {logits.shape}")  # (25, 21)
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

## Usage Modes

The new Equinox implementation supports three decoding modes:

### 1. Unconditional Decoding (Structure-Only)

```python
# Generate logits without sequence information
_, logits = model._call_unconditional(edge_features, neighbor_indices, mask)
```

### 2. Conditional Decoding (Known Sequence)

```python
# Score a known sequence
one_hot_sequence = jax.nn.one_hot(sequence_indices, 21)
ar_mask = jnp.ones((num_residues, num_residues))  # Full autoregressive mask

_, logits = model._call_conditional(
    edge_features,
    neighbor_indices,
    mask,
    ar_mask=ar_mask,
    one_hot_sequence=one_hot_sequence
)
```

### 3. Autoregressive Sampling

```python
# Sample a new sequence
temperature = jnp.array(0.1)
ar_mask = jnp.ones((num_residues, num_residues))

sampled_seq, final_logits = model._call_autoregressive(
    edge_features,
    neighbor_indices,
    mask,
    ar_mask=ar_mask,
    prng_key=key,
    temperature=temperature
)
```

## Functional Implementation (for comparison)

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

# Forward pass (multi-step)
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

print("Functional logits shape:", func_logits.shape)  # (25, 21)
```

## Verify Equivalence

```python
# Get outputs from both implementations
_, eqx_logits = model._call_unconditional(edge_features, neighbor_indices, mask)

# Functional path
node_features, edge_features_encoded = encoder_fn(edge_features, neighbor_indices, mask)
node_features_decoded = decoder_fn(node_features, edge_features_encoded, mask)
func_logits = final_projection(params, node_features_decoded)

# Check if outputs match
max_diff = jnp.max(jnp.abs(eqx_logits - func_logits))
mean_diff = jnp.mean(jnp.abs(eqx_logits - func_logits))
rel_diff = max_diff / (jnp.abs(func_logits).max() + 1e-9)

print(f"Max absolute difference: {max_diff:.2e}")
print(f"Max relative difference: {rel_diff:.2e}")
print(f"Mean difference: {mean_diff:.2e}")

# Check with tolerance
matches = jnp.allclose(eqx_logits, func_logits, rtol=1e-5, atol=1e-5)
print(f"Outputs match (rtol=1e-5, atol=1e-5): {matches}")
```

Expected output:

```text
Max absolute difference: ~1e-05
Max relative difference: ~1e-04
Mean difference: ~1e-06
Outputs match (rtol=1e-5, atol=1e-5): True
```

## Current Test Status

All core equivalence tests pass:

- ✅ Feature extraction (encoder): max_diff ~1e-06
- ✅ Unconditional decoder: max_diff ~1e-05
- ✅ Conditional decoder: max_diff ~1.26e-05
- ✅ Autoregressive first step: max_diff ~1e-05

Run tests:

```bash
pytest tests/test_eqx_equivalence.py -v
# 4 passed in 11.62s
```

## Save and Load Equinox Model

```python
import equinox

# Save model
equinox.tree_serialise_leaves("my_model.eqx", model)

# Load model (need to create empty structure first)
loaded_model = PrxteinMPNN.from_functional(
    params,
    num_encoder_layers=3,
    num_decoder_layers=3,
    key=key
)
loaded_model = equinox.tree_deserialise_leaves("my_model.eqx", loaded_model)

# Verify loaded model works
_, loaded_logits = loaded_model._call_unconditional(edge_features, neighbor_indices, mask)

# Should be bit-perfect identical
print(f"Save/load preserves output: {jnp.allclose(eqx_logits, loaded_logits, rtol=1e-7, atol=1e-8)}")
```

**Note**: Save/load preservation test not yet implemented. This will be added in Milestone 4.

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
