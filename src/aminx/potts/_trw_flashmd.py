"""Tiled reductions over axis-0 (FlashMD-style) for dense TRW message tensors.

Vendored pattern from jaxbeans ``physics/tiling.tile_reduction`` (scan + optional checkpoint)
to avoid a hard dependency on the sibling jaxbeans checkout on cluster nodes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def prod_pow_messages_axis0_tiled(
    m_current: Float[Array, "n n q"],
    rho: Float[Array, "n n"],
    *,
    tile_size: int,
    rematerialize: bool,
) -> Float[Array, "n q"]:
    """Compute ``prod_j m_current[j,i,a] ** rho[j,i]`` over ``j`` (axis 0) in tiles.

    Requires ``n % tile_size == 0``. Peak workspace scales like ``O(tile_size * n * q)``.
    """
    n, n2, q = m_current.shape
    if n2 != n:
        msg = f"expected square message tensor, got {(n, n2, q)}"
        raise ValueError(msg)
    if n % tile_size != 0:
        msg = f"n={n} must be divisible by tile_size={tile_size}"
        raise ValueError(msg)
    n_tiles = n // tile_size
    m_pow = jnp.power(m_current, rho[..., None])

    def tile_fn(j_idx: Array) -> Array:
        j0 = j_idx * tile_size
        slab = jax.lax.dynamic_slice_in_dim(m_pow, j0, tile_size, axis=0)
        return jnp.prod(slab, axis=0)

    if rematerialize:
        tile_fn = jax.checkpoint(tile_fn)

    def scan_step(carry: Array, j_idx: Array) -> tuple[Array, None]:
        return carry * tile_fn(j_idx), None

    init = jnp.ones((n, q), dtype=m_current.dtype)
    return jax.lax.scan(scan_step, init, jnp.arange(n_tiles, dtype=jnp.int32))[0]
