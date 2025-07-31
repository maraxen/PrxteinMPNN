"""Utilities for decoding order generation."""

from collections.abc import Callable

import jax
from jaxtyping import PRNGKeyArray

from .types import (
  DecodingOrder,
)

DecodingOrderInputs = tuple[PRNGKeyArray, int]
DecodingOrderOutputs = tuple[DecodingOrder, PRNGKeyArray]
DecodingOrderFn = Callable[[PRNGKeyArray, int], DecodingOrderOutputs]


@jax.jit
def random_decoding_order(
  prng_key: PRNGKeyArray,
  num_residues: int,
) -> DecodingOrderOutputs:
  """Return a random decoding order."""
  current_key, next_key = jax.random.split(prng_key)
  decoding_order = jax.random.permutation(current_key, jax.numpy.arange(num_residues))
  decoding_order = jax.numpy.asarray(decoding_order, dtype=jax.numpy.int32)
  return decoding_order, next_key
