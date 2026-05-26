"""Carry-bearing scan primitive — sibling to safe_map.

safe_map: stateless, no carry  →  (f, xs) → ys
safe_scan: stateful, with carry → (f, xs, init) → (final_carry, ys)

These are intentionally kept separate. safe_map's "no carry" contract is
load-bearing in kernel_dispatch.py and must not be overloaded.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import jax

if TYPE_CHECKING:
    from collections.abc import Callable

C = TypeVar("C")


def safe_scan(
    f: "Callable[[C, Any], tuple[C, Any]]",
    xs: Any,
    *,
    init: C,
) -> "tuple[C, Any]":
    """Apply carry-bearing scan over the leading axis of xs.

    Unlike safe_map, this is stateful: f receives and updates carry at each step.
    Wraps jax.lax.scan directly — no chunking variant (use safe_map + a
    SafeMap strategy for chunked stateless iteration).

    Args:
        f: Transition function (carry, x) -> (carry, y).
           carry must have static shape at JAX trace time.
        xs: Input pytree; leading axis is the scanned dimension.
            All leaves must share the same leading axis size.
        init: Initial carry value. May contain JAX arrays (traced leaves).
              Shape must be static.

    Returns:
        (final_carry, stacked_ys) where stacked_ys is a pytree with the same
        structure as the output y, stacked over the scanned axis.

    Raises:
        ValueError: If xs is an empty pytree.
    """
    leaves = jax.tree_util.tree_leaves(xs)
    if not leaves:
        msg = "xs must not be an empty PyTree"
        raise ValueError(msg)
    return jax.lax.scan(f, init, xs)


__all__ = ["safe_scan"]
