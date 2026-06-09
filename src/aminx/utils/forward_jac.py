"""Forward-mode categorical Jacobian utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aminx.sampling.conditional_logits import make_encoding_conditional_logits_split_fn
from aminx.types.arrays import CategoricalJacobian


def make_categorical_jacobian_fn(model):
  """Build a JIT function computing the (L, 21, L, 21) categorical Jacobian."""
  encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)

  @jax.jit
  def compute_categorical_jacobian(
    prng_key: jax.Array,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    sequence: jax.Array,
    *,
    backbone_noise: float = 0.0,
  ) -> CategoricalJacobian:
    encoding = encode_fn(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise=backbone_noise,
      prng_key=prng_key,
      structure_mapping=None,
    )
    if sequence.ndim == 1:
      one_hot = jax.nn.one_hot(sequence, 21)
    else:
      one_hot = sequence

    length = one_hot.shape[0]
    ar_mask = jnp.zeros((length, length), dtype=jnp.int32)

    def logits_from_one_hot(oh: jax.Array) -> jax.Array:
      return decode_fn(encoding, oh, ar_mask=ar_mask)

    return jax.jacfwd(logits_from_one_hot)(one_hot)

  return compute_categorical_jacobian
