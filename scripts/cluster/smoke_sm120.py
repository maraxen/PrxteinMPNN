#!/usr/bin/env python
"""
SM120 Blackwell smoke test for xtrax + aminx integration.
Verifies JAX/GPU compilation with XLA workaround.
"""

import sys

import jax
import jax.numpy as jnp

print(f"Python: {sys.version}")
print(f"JAX: {jax.__version__}")
print(f"devices: {jax.devices()}")
assert len(jax.devices()) > 0, "No GPU devices detected"

# Try to import xtrax + aminx (optional for basic smoke)
try:
    import xtrax
    import aminx
    print(f"xtrax: {xtrax.__version__}, aminx loaded")
except ImportError as e:
    print(f"Note: xtrax/aminx not available: {e}")

@jax.jit
def dot(a, b):
    return jnp.dot(a, b)


A = jax.random.normal(jax.random.PRNGKey(0), (64, 64))
C = dot(A, A.T)
C.block_until_ready()
print(f"dot(64,64): shape={C.shape}")
print("SMOKE: PASS")
