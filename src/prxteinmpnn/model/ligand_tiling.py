"""L-axis tiling helpers for ligand pairwise features (Jaxbeans-style inlined).

Adapted conceptually from FlashMD tiling: sequential scan over fixed axis-0 slabs,
optional jax.checkpoint per slab so peak intermediate memory scales with the slab size
rather than full L. No external jaxbeans dependency.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def pad_axis_to_multiple(arr: jax.Array, tile: int, axis: int = 0, *, pad_val: float = 0.0) -> jax.Array:
  """Pad `arr` so its length along `axis` is a multiple of `tile` (>0)."""
  if tile <= 0:
    raise ValueError("tile must be positive")

  dim = arr.shape[axis]
  rem = dim % tile
  n_pad = jax.lax.select(rem == 0, 0, tile - rem)
  widths = [(0, 0)] * arr.ndim
  widths[axis] = (0, n_pad)
  pad_const = jnp.asarray(pad_val, dtype=arr.dtype)
  return jnp.pad(arr, widths, constant_values=pad_const)


def map_chunks_axis0(
  y: jax.Array,
  *,
  chunk_size: int,
  fn: Callable[[jax.Array], jax.Array],
  use_checkpoint: bool = False,
) -> jax.Array:
  """Sequential axis-0 map: concatenate `fn` outputs on slabs, trim padding.

  `y` has shape (L, *rest). `fn` maps a slab (chunk_size, *rest_same) → (chunk_size, *rest_out).

  Axis 0 is padded up to the next multiple of `chunk_size` with zeros before scanning; the padded
  tail is removed from axis 0 of the flattened result via `slice` to restore length L.

  Sequential `scan` evaluates one slab at a time (helps GPU peak allocator pressure vs one giant GEMM).
  """
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")

  nd = y.ndim
  L = y.shape[0]
  rest_dims = tuple(y.shape[d] for d in range(1, nd))

  rem = L % chunk_size
  n_pad = jax.lax.select(rem == 0, 0, chunk_size - rem)
  widths = [(0, 0)] * nd
  widths[0] = (0, n_pad)
  y_pad = jnp.pad(y, widths, constant_values=jnp.asarray(0.0, dtype=y.dtype))

  n_chunks = y_pad.shape[0] // chunk_size

  f = jax.checkpoint(fn) if use_checkpoint else fn

  def _step(_carry: None, ii: jax.Array) -> tuple[None, jax.Array]:
    del _carry
    start = ii * chunk_size
    slc = jax.lax.dynamic_slice(y_pad, (start,) + (0,) * (nd - 1), (chunk_size,) + rest_dims)
    return None, f(slc)

  _, slabs = jax.lax.scan(_step, None, jnp.arange(n_chunks, dtype=jnp.int32))

  out_shape = slabs.shape  # (n_chunks, chunk_size, *_)
  fused_L_dim = out_shape[0] * out_shape[1]
  out_flat = jnp.reshape(slabs, (fused_L_dim,) + out_shape[2:])
  return out_flat[:L]
