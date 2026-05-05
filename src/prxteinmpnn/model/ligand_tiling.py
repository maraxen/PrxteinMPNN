"""L-axis tiling helpers for ligand pairwise features (Jaxbeans-style inlined).

Adapted conceptually from FlashMD tiling: sequential slabs along axis 0;
``scan`` carries a single output buffer updated via ``dynamic_update_slice``
(no stacked ys), so we never materialize all chunk outputs simultaneously.
No external jaxbeans dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

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
  """Sequential axis-0 slabs (JIT-friendly): fixed ``(chunk_size-1)`` tail pad + ``fori_loop``.

  `y` has shape (L, *rest). `fn` maps a slab (chunk_size, *rest_same) → (chunk_size, *rest_out).

  Each slab is written into one output buffer via ``dynamic_update_slice``. We avoid ``jnp.pad``
  with a traced width (which breaks ``jax.jit``) by prepending a **constant** tail slack of
  ``chunk_size - 1`` rows, enough to always ``dynamic_slice`` full ``chunk_size`` windows.

  Note: jax.checkpoint/remat around `fn` is not wrapped here; nested remat inside a loop body can
  hit TracerBoolConversionError with equinox Modules that use Python control flow on parameters.
  """
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")

  cs = chunk_size
  slack = cs - 1
  nd = y.ndim
  L = y.shape[0]
  rest_dims = tuple(y.shape[d] for d in range(1, nd))

  widths = [(0, slack)] + [(0, 0)] * (nd - 1)
  y_ext = jnp.pad(y, widths, constant_values=jnp.asarray(0, dtype=y.dtype))

  def start_idx(ax0_off: jax.Array | int) -> tuple[jax.Array | int, ...]:
    tail = tuple(0 for _ in range(nd - 1))
    return (ax0_off,) + tail

  tmpl = jax.eval_shape(
    fn,
    jax.lax.dynamic_slice(y_ext, start_idx(0), (cs,) + rest_dims),
  )

  buf = jnp.zeros((y_ext.shape[0],) + tmpl.shape[1:], dtype=tmpl.dtype)
  zstart = lambda rank: tuple(0 for _ in range(rank - 1))

  def body(ii: jax.Array, acc: jax.Array) -> jax.Array:
    start = ii * cs
    slc = jax.lax.dynamic_slice(y_ext, start_idx(start), (cs,) + rest_dims)
    out_slab = fn(slc)
    return jax.lax.dynamic_update_slice(acc, out_slab, (start,) + zstart(tmpl.ndim))

  n_chunks_i32 = jnp.maximum(
    jnp.int32(0),
    (jnp.asarray(L, dtype=jnp.int32) + jnp.int32(cs) - jnp.int32(1)) // jnp.int32(cs),
  )
  filled = jax.lax.fori_loop(jnp.int32(0), n_chunks_i32, body, buf)
  slice_sizes = (L,) + tmpl.shape[1:]
  slice_zeros = tuple(0 for _ in range(tmpl.ndim))
  return jax.lax.dynamic_slice(filled, slice_zeros, slice_sizes)


def map_chunks_axis0_multi(
  fn: Callable[..., tuple[jax.Array, ...]],
  chunk_size: int,
  arrays_in: Sequence[jax.Array],
) -> tuple[jax.Array, ...]:
  """Sequential axis-0 slabs: ``fn(*slabs) -> outputs``; JIT-friendly slack pad + ``fori_loop``.

  Inputs share axis-0 ``L``. Each receives a fixed tail slack of ``chunk_size - 1`` zeros (constant
  ``jnp.pad`` width) so slabs never use traced paddings. Outputs are trimmed back to ``L``.
  """
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")

  arrays_in_tuple = tuple(arrays_in)
  if not arrays_in_tuple:
    raise ValueError("arrays_in must be non-empty")

  cs = chunk_size
  slack = cs - 1

  L_ax = arrays_in_tuple[0].shape[0]
  for a in arrays_in_tuple[1:]:
    if a.shape[0] != L_ax:
      raise ValueError("map_chunks_axis0_multi: mismatched axis-0 lengths")

  arrays_ext: list[jax.Array] = []
  for arr in arrays_in_tuple:
    w = [(0, slack)] + [(0, 0)] * (arr.ndim - 1)
    arrays_ext.append(jnp.pad(arr, w, constant_values=jnp.asarray(0, dtype=arr.dtype)))

  slabs0 = tuple(
    jax.lax.dynamic_slice(
      ap,
      (0,) + (0,) * (ap.ndim - 1),
      (cs,) + tuple(ap.shape[d] for d in range(1, ap.ndim)),
    )
    for ap in arrays_ext
  )
  out_shapes = jax.eval_shape(fn, *slabs0)
  outs_tpl = tuple(out_shapes) if isinstance(out_shapes, tuple) else (out_shapes,)

  Lex = arrays_ext[0].shape[0]
  out_bufs = tuple(
    jnp.zeros((Lex,) + tuple(sh.shape[d] for d in range(1, sh.ndim)), dtype=sh.dtype) for sh in outs_tpl
  )

  def zstart(rank: int) -> tuple[int, ...]:
    return tuple(0 for _ in range(rank - 1))

  def body(ii: jax.Array, acc: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    start = ii * cs
    slabs_in = tuple(
      jax.lax.dynamic_slice(
        ap,
        (start,) + (0,) * (ap.ndim - 1),
        (cs,) + tuple(ap.shape[d] for d in range(1, ap.ndim)),
      )
      for ap in arrays_ext
    )
    out_slabs = fn(*slabs_in)
    return tuple(
      jax.lax.dynamic_update_slice(buf, osl, (start,) + zstart(osl.ndim))
      for buf, osl in zip(acc, out_slabs, strict=True)
    )

  n_chunks_i32 = jnp.maximum(
    jnp.int32(0),
    (jnp.asarray(L_ax, dtype=jnp.int32) + jnp.int32(cs) - jnp.int32(1)) // jnp.int32(cs),
  )
  filled = jax.lax.fori_loop(jnp.int32(0), n_chunks_i32, body, out_bufs)

  return tuple(
    jax.lax.dynamic_slice(
      buf,
      (0,) + (0,) * (buf.ndim - 1),
      (L_ax,) + tuple(buf.shape[d] for d in range(1, buf.ndim)),
    )
    for buf in filled
  )
