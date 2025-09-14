"""Utilities for Average Product Correction (APC) and related operations."""

import jax
import jax.numpy as jnp


def mean_center(jacobian: jax.Array) -> jax.Array:
  """Mean-centers a 4D array representing pairwise interactions."""
  mean_dim3 = jnp.mean(jacobian, axis=3, keepdims=True)
  mean_dim1 = jnp.mean(jacobian, axis=1, keepdims=True)
  total_mean = jnp.mean(jacobian, keepdims=True)
  return jacobian - mean_dim1 - mean_dim3 + total_mean


def symmetrize(jacobian: jax.Array) -> jax.Array:
  """Symmetrize a 4D array representing pairwise interactions."""
  return 0.5 * (jacobian + jnp.transpose(jacobian, (2, 3, 0, 1)))


def calculate_frobenius_norm_per_pair(sym_jacobian: jax.Array) -> jax.Array:
  """Calculate the Frobenius norm for each residue pair in a 4D symmetric array."""
  protein_len = sym_jacobian.shape[0]

  def frobenius_for_pair(i: int, j: int) -> jax.Array:
    return jnp.linalg.norm(sym_jacobian[i, :, j, :], ord="fro")

  return jax.vmap(lambda i: jax.vmap(lambda j: frobenius_for_pair(i, j))(jnp.arange(protein_len)))(
    jnp.arange(protein_len),
  )


def apc_correction(frobenius_matrix: jax.Array) -> jax.Array:
  """Apply Average Product Correction (APC) to a Frobenius norm matrix."""
  row_means = jnp.mean(frobenius_matrix, axis=1, keepdims=True)
  col_means = jnp.mean(frobenius_matrix, axis=0, keepdims=True)
  total_mean = jnp.mean(frobenius_matrix)
  apc_matrix = row_means * col_means / total_mean
  return frobenius_matrix - apc_matrix


def apc_corrected_frobenius_norm(jacobian: jax.Array) -> jax.Array:
  """Compute the APC-corrected Frobenius norm from a 4D Jacobian array."""
  sym_jacobian = symmetrize(jacobian)
  centered_jacobian = mean_center(sym_jacobian)
  frobenius_matrix = calculate_frobenius_norm_per_pair(centered_jacobian)
  return apc_correction(frobenius_matrix)
