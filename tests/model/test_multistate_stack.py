"""Tests for scatter / gather helpers on stacked multistate logits."""

import jax
import jax.numpy as jnp

from prxteinmpnn.model.multistate_stack import gather_flat_to_stack, scatter_stack_to_flat


def test_scatter_gather_roundtrip_fixed_injective():
  S, P, C, n_flat = 2, 4, 7, 8
  key = jax.random.PRNGKey(0)
  stack = jax.random.normal(key, (S, P, C))
  # Distinct nonnegative flat indices filling 8 spots (some stack pad -1 skipped)
  rows = jnp.array(
    [
      [0, 1, 2, 3],
      [4, 5, 6, 7],
    ],
    dtype=jnp.int32,
  )
  logits_f = scatter_stack_to_flat(stack, rows, n_flat)
  back = gather_flat_to_stack(logits_f, rows)
  assert jax.device_get(back.shape) == (S, P, C)
  assert jnp.allclose(back, stack, rtol=1e-6, atol=1e-6)


def test_scatter_skip_negative_rows():
  n_flat = 8
  stack = jnp.ones((2, 3, 4), dtype=jnp.float32)
  rows = jnp.array([[-1, 0, 1], [2, -1, 7]], dtype=jnp.int32)
  out = scatter_stack_to_flat(stack, rows, n_flat)
  assert out.shape == (n_flat, 4)
  v = jax.device_get(out)
  assert (v[0] == 1).all()
  assert (v[7] == 1).all()
  assert jnp.all(out[6] == 0)


def test_gather_masks_invalid_below_zero():
  n_flat = 3
  flat = jax.random.normal(jax.random.key(7), (n_flat, 5))
  rows = jnp.array([[0], [-1]], dtype=jnp.int32)
  g = gather_flat_to_stack(flat, rows)
  delta = jax.device_get(jnp.abs(g[0, 0] - flat[0]).max())
  assert delta < 1e-7
  assert float(jax.device_get(jnp.linalg.norm(g[1]))) == 0.0
