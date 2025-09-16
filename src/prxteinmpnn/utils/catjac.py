"""Utilities for computing categorical Jacobians."""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp

from prxteinmpnn.utils.align import align_sequences
from prxteinmpnn.utils.types import CategoricalJacobian, InterproteinMapping, ProteinSequence

CombineCatJacPairFn = Callable[
  [
    CategoricalJacobian,  # Jacobian for protein A
    CategoricalJacobian,  # Jacobian for protein B (to be mapped)
    InterproteinMapping,  # The (L, 2) index map from B to A
    jax.Array,  # The scalar weight for the pair
  ],
  CategoricalJacobian,  # The combined Jacobian
]

# The public-facing function that operates on the whole batch
CombineCatJacFn = Callable[
  [
    jax.Array,  # Full batch of Jacobians, shape (N, ...)
    ProteinSequence,  # Full batch of sequences, shape (N, L)
    jax.Array | None,  # Per-protein weights, shape (N,)
  ],
  jax.Array,  # Final combined Jacobians, shape (N*(N-1)/2, ...)
]

# The factory that creates the public-facing function
CombineCatJacFnFactory = Callable[
  [
    CombineCatJacPairFn | Literal["add", "subtract"],
  ],
  CombineCatJacFn,
]


def _gather_mapped_jacobian(
  jacobian_to_map: CategoricalJacobian,
  mapping: jax.Array,
) -> CategoricalJacobian:
  """Remaps a Jacobian using an index map from an alignment."""
  k_indices = mapping[:, 1]
  valid_mask = k_indices != -1
  k_indices_clipped = jnp.maximum(0, k_indices)

  mapped_jac = jacobian_to_map[:, k_indices_clipped][:, :, :, k_indices_clipped]

  mask_i = valid_mask[:, None, None, None]
  mask_j = valid_mask[None, None, :, None]

  return mapped_jac * mask_i * mask_j


def _add_jacobians_mapped(
  jac1: CategoricalJacobian,
  jac2: CategoricalJacobian,
  mapping: jax.Array,
  weights: jax.Array,
) -> CategoricalJacobian:
  """Add jac1 to the mapped version of jac2."""
  mapped_jac2 = _gather_mapped_jacobian(jac2, mapping)
  return jac1 + mapped_jac2 * weights


def _subtract_jacobians_mapped(
  jac1: CategoricalJacobian,
  jac2: CategoricalJacobian,
  mapping: jax.Array,
  weights: jax.Array,
) -> CategoricalJacobian:
  """Subtracts the mapped version of jac2 from jac1."""
  mapped_jac2 = _gather_mapped_jacobian(jac2, mapping)
  return jac1 - mapped_jac2 * weights


_MIN_NUMBER_PROTEINS = 2


def make_combine_jac(
  combine_fn: Literal["add", "subtract"] | CombineCatJacPairFn,  # Simplified for clarity
  fn_kwargs: dict[str, Any] | None = None,
  batch_size: int | None = None,
) -> CombineCatJacFn:
  """Create a function to combine Jacobians using a specified operation."""
  if combine_fn == "add":
    _combine_fn = _add_jacobians_mapped
  elif combine_fn == "subtract":
    _combine_fn = _subtract_jacobians_mapped
  else:
    _combine_fn = partial(combine_fn, **(fn_kwargs or {}))

  def combine_jacobians(
    jacobians: jax.Array,
    sequences: ProteinSequence,
    weights: jax.Array | None = None,
  ) -> jax.Array:
    """Compute cross-protein Jacobian combinations for all pairs."""
    if weights is None:
      weights = jnp.ones((jacobians.shape[0],), dtype=jacobians.dtype)
    n_proteins = jacobians.shape[0]
    if n_proteins < _MIN_NUMBER_PROTEINS:
      return jnp.empty((0, *jacobians.shape[1:]))

    i_indices, j_indices = jnp.triu_indices(n_proteins, k=1)

    jac_i = jacobians[i_indices]
    jac_j = jacobians[j_indices]
    pair_weights = weights[i_indices]
    all_mappings = align_sequences(sequences)

    xs = (jac_i, jac_j, all_mappings, pair_weights)

    return jax.lax.map(
      lambda x: _combine_fn(*x),
      xs,
      batch_size=batch_size,
    )

  return combine_jacobians
