"""Utilities for decoding order generation.

prxteinmpnn.utils.decoding_order
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray

from .types import (
  DecodingOrder,
)

DecodingOrderInputs = tuple[PRNGKeyArray, Int, Array]
DecodingOrderOutputs = tuple[DecodingOrder, PRNGKeyArray]
DecodingOrderFn = Callable[[*DecodingOrderInputs], DecodingOrderOutputs]


def get_decoding_order(
    key: PRNGKeyArray, tie_group_map: jnp.ndarray
) -> DecodingOrder:
    """Generate a decoding order over tie groups."""
    unique_group_ids = jnp.unique(tie_group_map)
    group_decoding_order = jax.random.permutation(key, unique_group_ids)
    return group_decoding_order


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
