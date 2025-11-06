"""Utilities for decoding order generation.

prxteinmpnn.utils.decoding_order
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Int, PRNGKeyArray

from .autoregression import get_decoding_step_map
from .types import (
  DecodingOrder,
)

DecodingOrderInputs = tuple[PRNGKeyArray, Int, jnp.ndarray | None]  # Added tie_group_map
DecodingOrderOutputs = tuple[DecodingOrder, PRNGKeyArray]
DecodingOrderFn = Callable[
  [PRNGKeyArray, int, jnp.ndarray | None, int | None],
  DecodingOrderOutputs,
]


@partial(jax.jit, static_argnames=("num_residues", "num_groups"))
def random_decoding_order(
  prng_key: PRNGKeyArray,
  num_residues: int,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
) -> DecodingOrderOutputs:
  """Return a random decoding order, optionally respecting tied positions.

  Args:
    prng_key: PRNG key for randomness.
    num_residues: Total number of residues.
    tie_group_map: Optional (N,) array mapping each position to a group ID.
                   Positions with the same group ID are tied and will be
                   decoded together in the same step.
    num_groups: Number of unique groups in tie_group_map. Required if
                tie_group_map is provided. Should equal tie_group_map.max() + 1
                when groups are normalized to [0, 1, ..., num_groups-1].

  Returns:
    Tuple of (decoding_order, next_key) where decoding_order respects ties.

  Example:
    >>> key = jax.random.PRNGKey(0)
    >>> # Without ties: standard random order
    >>> order, key = random_decoding_order(key, 5)
    >>>
    >>> # With ties: positions in same group stay together
    >>> tie_map = jnp.array([0, 1, 0, 2, 1])  # Groups: {0: [0,2], 1: [1,4], 2: [3]}
    >>> order, key = random_decoding_order(key, 5, tie_map, num_groups=3)

  """
  current_key, next_key = jax.random.split(prng_key)

  if tie_group_map is None:
    # Standard random order without ties
    decoding_order = jax.random.permutation(current_key, jnp.arange(0, num_residues))
    return jnp.asarray(decoding_order, dtype=jnp.int32), next_key

  if num_groups is None:
    msg = "num_groups must be provided when tie_group_map is not None"
    raise ValueError(msg)

  # With tied positions: generate random order over groups (vectorized)
  group_order = jax.random.permutation(current_key, jnp.arange(num_groups))

  # Map groups to decoding steps
  decoding_step_map = get_decoding_step_map(tie_group_map, group_order, num_groups)

  # Create decoding order: sort by step, then by position within step
  # This ensures positions in the same group are adjacent
  decoding_order = jnp.argsort(decoding_step_map)

  return jnp.asarray(decoding_order, dtype=jnp.int32), next_key


def single_decoding_order(
  key: PRNGKeyArray,
  num_residues: int,
  tie_group_map: jnp.ndarray | None = None,
  num_groups: int | None = None,
) -> DecodingOrderOutputs:
  """Generate a single decoding order (identity permutation).

  Args:
      key: Random key (unused, for API compatibility).
      num_residues: Number of residues.
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          Currently ignored for single decoding order.
      num_groups: Number of unique groups (unused, for API compatibility).

  Returns:
      decoding_order: (N,) array, [0, 1, ..., N-1].
      key: Same random key (unchanged).

  """
  del tie_group_map, num_groups  # Unused for single order
  decoding_order = jnp.arange(0, num_residues, dtype=jnp.int32)
  return decoding_order, key
