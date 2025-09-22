"""Utilities for Average Product Correction (APC) and related operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import CategoricalJacobian

# Constants
JACOBIAN_EXPECTED_NDIM = 4


def mean_center(jacobian: CategoricalJacobian) -> CategoricalJacobian:
  """Mean-centers a 4D array representing pairwise interactions."""
  mean_dim3 = jnp.mean(jacobian, axis=3, keepdims=True)
  mean_dim1 = jnp.mean(jacobian, axis=1, keepdims=True)
  total_mean = jnp.mean(jacobian, keepdims=True)
  return jacobian - mean_dim1 - mean_dim3 + total_mean


def symmetrize(jacobian: CategoricalJacobian) -> CategoricalJacobian:
  """Symmetrize a 4D array representing pairwise interactions."""
  return 0.5 * (jacobian + jnp.transpose(jacobian, (2, 3, 0, 1)))


def calculate_frobenius_norm_per_pair(
  sym_jacobian: CategoricalJacobian,
  residue_batch_size: int = 1000,
) -> jax.Array:
  """Calculate the Frobenius norm for each (i, j) residue pair's interaction matrix.

  This function operates on a Jacobian of shape (L, K, L, K) and returns an
  (L, L) matrix of norms. It is designed to be memory-efficient by mapping
  over the residue dimensions rather than creating a large intermediate
  transposed array.

  Args:
    sym_jacobian: A 4D array representing the Jacobian, with shape
      (L, K, L, K), where L is the sequence length and K is the number of
      features (e.g., amino acids).
    residue_batch_size: The number of residues to process in each batch. This
      parameter helps manage memory usage during the computation.

  Returns:
    An (L, L) array where each element (i, j) is the Frobenius norm of the
    corresponding (K, K) sub-matrix.

  """
  if sym_jacobian.ndim != JACOBIAN_EXPECTED_NDIM:
    sym_jacobian = jnp.squeeze(sym_jacobian, axis=0)

  def frobenius_for_residue_i(jacobian_i_slice: jax.Array) -> jax.Array:
    """Calculate norms for a single residue 'i' against all residues 'j'."""
    jacobian_i_transposed = jnp.transpose(jacobian_i_slice, (1, 0, 2))
    return jax.lax.map(jnp.linalg.norm, jacobian_i_transposed)

  return jax.lax.map(frobenius_for_residue_i, sym_jacobian, batch_size=residue_batch_size)


def apc_correction(frobenius_matrix: jax.Array) -> jax.Array:
  """Apply Average Product Correction (APC) to a Frobenius norm matrix."""
  row_means = jnp.mean(frobenius_matrix, axis=1, keepdims=True)
  col_means = jnp.mean(frobenius_matrix, axis=0, keepdims=True)
  total_mean = jnp.mean(frobenius_matrix)
  apc_matrix = row_means * col_means / total_mean
  return frobenius_matrix - apc_matrix


def apc_corrected_frobenius_norm(
  jacobian: CategoricalJacobian,
  residue_batch_size: int = 1000,
) -> CategoricalJacobian:
  """Compute the APC-corrected Frobenius norm from a 4D Jacobian array."""
  sym_jacobian = symmetrize(jacobian)
  centered_jacobian = mean_center(sym_jacobian)
  frobenius_matrix = calculate_frobenius_norm_per_pair(
    centered_jacobian,
    residue_batch_size=residue_batch_size,
  )
  return apc_correction(frobenius_matrix)
