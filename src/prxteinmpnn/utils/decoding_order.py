import jax.numpy as jnp


def get_decoding_order(tie_group_map: jnp.ndarray, key) -> jnp.ndarray:
  """Generate a decoding order over tie groups.

  Args:
    tie_group_map: (N,) array of group ids.
    key: PRNGKey for shuffling.

  Returns:
    group_decoding_order: (M,) array, permutation of unique group ids.

  """
  unique_group_ids = jnp.unique(tie_group_map)
  group_decoding_order = jax.random.permutation(key, unique_group_ids)
  return group_decoding_order


"""Utilities for decoding order generation.

prxteinmpnn.utils.decoding_order
"""

from collections.abc import Callable
from functools import partial

import jax
from jaxtyping import Int, PRNGKeyArray

from .types import (
  DecodingOrder,
)

DecodingOrderInputs = tuple[PRNGKeyArray, Int]
DecodingOrderOutputs = tuple[DecodingOrder, PRNGKeyArray]
DecodingOrderFn = Callable[[*DecodingOrderInputs], DecodingOrderOutputs]


@partial(jax.jit, static_argnames=("num_residues",))
def random_decoding_order(
  prng_key: PRNGKeyArray,
  num_residues: int,
) -> DecodingOrderOutputs:
  """Return a random decoding order."""
  current_key, next_key = jax.random.split(prng_key)
  decoding_order = jax.random.permutation(current_key, jax.numpy.arange(0, num_residues))
  decoding_order = jax.numpy.asarray(decoding_order, dtype=jax.numpy.int32)
  return decoding_order, next_key


@partial(jax.jit, static_argnames=("num_residues",))
def single_decoding_order(
  prng_key: PRNGKeyArray,
  num_residues: int,
) -> DecodingOrderOutputs:
  """Return a single decoding order (identity)."""
  decoding_order = jax.random.permutation(prng_key, jax.numpy.arange(0, num_residues))
  decoding_order = jax.numpy.asarray(decoding_order, dtype=jax.numpy.int32)
  return decoding_order, prng_key
