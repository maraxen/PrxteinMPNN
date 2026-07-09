"""Shared fixtures for aminx.ebm tests (backlog node E0)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def _toy_energy(w: jax.Array, x: jax.Array, mask: jax.Array) -> jax.Array:
  """Toy analog of the E3 ``EnergyReadout``: ``E(x) = sum_i mask_i * ||W @ x_i||**2``.

  Substitutes the diffused coordinates ``x`` directly for the (not-yet-
  implemented) trunk features ``a`` -- ``r_i = W @ x_i`` plays the role of
  the per-residue vector head output. Used to prove the E3 readout
  invariants harness (zero-energy, non-negativity, score = -grad(E), and
  the 2nd-order training-grad gate) independently of the pending model port.
  """
  r = jnp.einsum("od,nd->no", w, x)  # [N, D_out], r_i = W @ x_i
  per_residue = jnp.sum(r**2, axis=-1)  # [N]
  return jnp.sum(mask.astype(per_residue.dtype) * per_residue)


@pytest.fixture
def toy_energy_fn():
  """Return the toy energy ``E(x) = sum_i mask_i * ||W @ x_i||**2``."""
  return _toy_energy


@pytest.fixture
def enable_x64():
  """Enable ``jax_enable_x64`` for the duration of a test, then restore it.

  Used only by the 2nd-order finite-difference gate, where float32
  precision would swamp a central-difference epsilon.
  """
  previous = jax.config.jax_enable_x64
  jax.config.update("jax_enable_x64", True)
  try:
    yield
  finally:
    jax.config.update("jax_enable_x64", previous)
