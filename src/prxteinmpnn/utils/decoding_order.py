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
