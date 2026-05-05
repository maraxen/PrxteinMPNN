"""L-axis tiling helpers for ligand pairwise features (Jaxbeans-style inlined).

Adapted conceptually from FlashMD tiling: sequential slabs along axis 0;
``scan`` carries a single output buffer updated via ``dynamic_update_slice``
(no stacked ys), so we never materialize all chunk outputs simultaneously.
No external jaxbeans dependency.
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
) -> jax.Array:
  """Sequential axis-0 reduce into one buffer; trim padding on axis 0.

  `y` has shape (L, *rest). `fn` maps a slab (chunk_size, *rest_same) → (chunk_size, *rest_out).

  Axis 0 is padded to a multiple of `chunk_size`. Each slab is written into a dedicated output
  buffer via `dynamic_update_slice` inside `scan`; the scanned value is carry-only (`None` ys stack)
  so JAX does not keep all slabs resident at once.

  Note: jax.checkpoint/remat around `fn` is not wrapped here; nested remat inside a scan body can
  hit TracerBoolConversionError with equinox Modules that use Python control flow on parameters.
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

  template_slice = jax.lax.dynamic_slice(
    y_pad, (0,) + (0,) * (nd - 1), (chunk_size,) + rest_dims
  )
  tmpl = jax.eval_shape(fn, template_slice)
  zeros_tail = tuple(0 for _ in range(tmpl.ndim - 1))

  buf = jnp.zeros((y_pad.shape[0],) + tmpl.shape[1:], dtype=tmpl.dtype)

  def scan_body(acc: jax.Array, ii: jax.Array) -> tuple[jax.Array, None]:
    start = ii * chunk_size
    slc = jax.lax.dynamic_slice(y_pad, (start,) + (0,) * (nd - 1), (chunk_size,) + rest_dims)
    out_slab = fn(slc)
    return jax.lax.dynamic_update_slice(acc, out_slab, (start,) + zeros_tail), None

  filled = jax.lax.scan(scan_body, buf, jnp.arange(n_chunks, dtype=jnp.int32))[0]
  slice_sizes = (L,) + tmpl.shape[1:]
  slice_zeros = tuple(0 for _ in range(tmpl.ndim))
  return jax.lax.dynamic_slice(filled, slice_zeros, slice_sizes)
