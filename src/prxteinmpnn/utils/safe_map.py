"""Utility for safe mapping over arrays, avoiding XLA loop issues."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import jax

if TYPE_CHECKING:
  from collections.abc import Callable

T = TypeVar("T")
U = TypeVar("U")


def safe_map(
  f: Callable[[T], U],
  xs: Any,  # noqa: ANN401
  batch_size: int | None = None,
) -> Any:  # noqa: ANN401
  """
  Map a function over the first axis of xs.

  This function dispatches to jax.vmap if the input size is smaller than or equal to
  the batch size (or if batch_size is None), avoiding the overhead and potential
  XLA issues of jax.lax.map's loop construct for single-batch or small inputs.
  Otherwise, it falls back to jax.lax.map.

  Args:
      f: The function to map.
      xs: The input array(s).
      batch_size: The batch size for processing.

  Returns:
      The result of mapping f over xs.
  """
  leaves = jax.tree_util.tree_leaves(xs)
  if not leaves:
    raise ValueError("Input xs must not be an empty PyTree")
  
  num_elements = leaves[0].shape[0]
  
  if batch_size is None or num_elements <= batch_size:
    return jax.vmap(f)(xs)
  
  return jax.lax.map(f, xs, batch_size=batch_size)
