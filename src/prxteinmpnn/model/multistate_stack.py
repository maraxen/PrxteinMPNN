"""Stack ↔ flat indexing for ``state_vmap_exact`` logits (scatter / gather helpers)."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def scatter_stack_to_flat(
  logits_stack: jnp.ndarray,
  state_flat_rows: jnp.ndarray,
  n_flat: int,
  *,
  fill_value: float | None = None,
) -> jnp.ndarray:
  """Write stack rows into a flat logits buffer using ``state_flat_rows[s,j]`` as flat index.

  Args:
      logits_stack: ``(S, P, C)`` per-state logits including padding slots.
      state_flat_rows: ``(S, P)`` int; ``-1`` or out-of-range skips the write.
      n_flat: Length of supersystem logits (``N`` leading dim).
      fill_value: Unused (reserved); output starts at zeros.

  Returns:
      ``(n_flat, C)`` logits. If multiple stack slots map to the same flat row, the **last**
      slot in flattened ``(s,j)`` order wins (layout should be injective for parity with flat graphs).
  """
  del fill_value  # preserved for API symmetry with gather if we add masking later
  S, P, C = logits_stack.shape
  vals = jnp.reshape(logits_stack, (-1, C))
  idx = jnp.reshape(state_flat_rows, (-1,))
  n_step = vals.shape[0]
  logits = jnp.zeros((n_flat, C), dtype=logits_stack.dtype)

  def step(i: jnp.ndarray, acc: jnp.ndarray) -> jnp.ndarray:
    def _upd() -> jnp.ndarray:
      ri = jnp.int32(idx[i])
      vi = vals[i]
      return acc.at[ri].set(vi)

    def _noop() -> jnp.ndarray:
      return acc

    return jax.lax.cond(
      jnp.logical_and(idx[i] >= 0, idx[i] < jnp.int32(n_flat)),
      _upd,
      _noop,
    )

  return jax.lax.fori_loop(0, jnp.int32(n_step), step, logits)


def gather_flat_to_stack(
  logits_flat: jnp.ndarray,
  state_flat_rows: jnp.ndarray,
) -> jnp.ndarray:
  """Gather flat rows into ``(S, P, C)``. Invalid ``state_flat_rows`` become zeros."""
  lf = jnp.asarray(logits_flat, logits_flat.dtype)
  rows = jnp.asarray(state_flat_rows, jnp.int32)
  n_flat = jnp.int32(lf.shape[0])
  clipped = jnp.clip(rows, jnp.int32(0), jnp.maximum(n_flat - jnp.int32(1), jnp.int32(0)))
  out = jnp.take(lf, clipped, axis=0)
  invalid = jnp.logical_or(rows < 0, rows >= jnp.int32(n_flat))
  return jnp.where(invalid[..., jnp.newaxis], jnp.zeros_like(out), out)
