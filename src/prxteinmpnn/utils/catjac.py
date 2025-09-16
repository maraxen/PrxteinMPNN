"""Utilities for computing categorical Jacobians."""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp

from prxteinmpnn.utils.align import align_sequences
from prxteinmpnn.utils.types import CategoricalJacobian, InterproteinMapping, ProteinSequence

CombineCatJacFn = Callable[
  [CategoricalJacobian, ProteinSequence, jax.Array | None],
  CategoricalJacobian,
]
CombineCatJacTranformFn = (
  Callable[
    [CategoricalJacobian, InterproteinMapping, jax.Array | None, Any],
    CategoricalJacobian,
  ]
  | Callable[
    [CategoricalJacobian, InterproteinMapping, jax.Array | None],
    CategoricalJacobian,
  ]
)
CombineCatJacFnFactory = Callable[
  [CombineCatJacTranformFn | Literal["add", "subtract"]],
  CombineCatJacFn,
]


def add_jacobians(
  jacobians: CategoricalJacobian,
  mapping: InterproteinMapping,
  weights: jax.Array | None = None,
) -> CategoricalJacobian:
  """Combine two Jacobian tensors by summing them.

  Args:
    jacobians: Jacobian tensor of shape (batch_size, noise_levels, L, 21, L, 21)
    mapping: Integer mapping array of shape (batch_size, L) indicating residue correspondences.
    weights: Optional weights for each protein in the batch, shape (batch_size,)

  Returns:
    Combined Jacobian tensor of the same shape as inputs.

  """
  return jacobians + jnp.einsum(
    "b n i a j c, b i i' -> b n i' a j c",
    jacobians,
    mapping,
  ) * (weights[:, None, None, None, None, None] if weights is not None else 1.0)


def subtract_jacobians(
  jacobians: CategoricalJacobian,
  mapping: InterproteinMapping,
  weights: jax.Array | None = None,
) -> CategoricalJacobian:
  """Compute the difference between two Jacobian tensors.

  Args:
    jacobians: Jacobian tensor of shape (batch_size, noise_levels, L, 21, L, 21)
    mapping: Integer mapping array of shape (batch_size, L) indicating residue correspondences.
    weights: Optional weights for each protein in the batch, shape (batch_size,)

  Returns:
    Difference Jacobian tensor of the same shape as inputs.

  """
  return jacobians - jnp.einsum(
    "b n i a j c, b i i' -> b n i' a j c",
    jacobians,
    mapping,
  ) * (weights[:, None, None, None, None, None] if weights is not None else 1.0)


def make_combine_jac(
  combine_fn: CombineCatJacTranformFn | Literal["add", "subtract"],
  fn_kwargs: dict[str, Any] | None = None,
  batch_size: int | None = None,
) -> CombineCatJacFn:
  """Create a function to combine Jacobians using a specified operation.

  Args:
    combine_fn: A function that takes two Jacobian tensors and combines them
      (e.g., add_jacobians or subtract_jacobians).
    fn_kwargs: Optional keyword arguments for the combine function.
    batch_size: Optional batch size for processing pairs of proteins.

  Returns:
    A function that takes Jacobians, sequences, and weights, and applies the combination operation.

  """
  if isinstance(combine_fn, str):
    if combine_fn == "add":
      combine_fn = add_jacobians
    elif combine_fn == "subtract":
      combine_fn = subtract_jacobians
    else:
      msg = f"Invalid combine_fn string: {combine_fn}"
      raise ValueError(msg)

  _combine_fn = partial(combine_fn, **fn_kwargs) if fn_kwargs is not None else combine_fn

  def combine_jacobians(
    jacobians: jax.Array,
    sequences: ProteinSequence,
    weights: jax.Array | None = None,
  ) -> jax.Array:
    """Compute cross-protein Jacobian combinations for all pairs.

    Args:
      jacobians: Jacobian tensors of shape (N, noise_levels, L, 21, L, 21)
      sequences: Integer-encoded sequences of shape (N, L)
      weights: Optional weights for each protein in the batch, shape (N,)

    Returns:
      Combined Jacobians for all pairs.
      Shape: (N*(N-1), noise_levels, L, 21, L, 21)

    """
    n_proteins = jacobians.shape[0]
    if n_proteins < 2:
      return jnp.empty((0, *jacobians.shape[1:]))

    # Generate all pairs of indices (i, j) where i != j
    idx = jnp.arange(n_proteins)
    i, j = jnp.meshgrid(idx, idx)
    i, j = i.flatten(), j.flatten()
    mask = i != j
    i, j = i[mask], j[mask]

    # Gather data for each pair
    jac_i = jax.tree_util.tree_map(lambda x: x[i], jacobians)
    seq_i = jax.tree_util.tree_map(lambda x: x[i], sequences)
    seq_j = jax.tree_util.tree_map(lambda x: x[j], sequences)
    weights_i = jax.tree_util.tree_map(lambda x: x[i], weights) if weights is not None else None

    # Align sequences for each pair
    # align_sequences expects (batch, L), so we vmap it over the pairs
    mapping = jax.vmap(align_sequences)(jnp.stack([seq_i, seq_j], axis=1))

    # Combine Jacobians for each pair
    return jax.lax.map(
      _combine_fn,
      (jac_i, mapping, weights_i),
      batch_size=batch_size,
    )

  return combine_jacobians
