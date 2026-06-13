#!/usr/bin/env python
"""
SM120 Blackwell smoke test for xtrax + aminx integration.
Verifies JAX/GPU/xtrax import and basic JIT compilation.
"""

import sys

import jax
import jax.numpy as jnp
import xtrax
import aminx

print(f"Python: {sys.version}")
print(f"JAX: {jax.__version__}, xtrax: {xtrax.__version__}")
print(f"devices: {jax.devices()}")
assert len(jax.devices()) > 0, "No GPU devices detected"


@jax.jit
def dot(a, b):
    return jnp.dot(a, b)


A = jax.random.normal(jax.random.PRNGKey(0), (64, 64))
C = dot(A, A.T)
C.block_until_ready()
print(f"dot(64,64): shape={C.shape}")
print("SMOKE: PASS")
