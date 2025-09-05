"""Core functionalities for Protein Strain Analysis (PSA).

This module provides functions to compute the deformation gradient, Green-Lagrange strain tensor,
and principal strains from protein structures. The calculations are implemented in JAX for high performance.
"""

import jax.numpy as jnp
from jax import jit

from prxteinmpnn.utils.data_structures import ProteinStructure


@jit
def calculate_deformation_gradient(
  reference_coordinates: jnp.ndarray,
  deformed_coordinates: jnp.ndarray,
) -> jnp.ndarray:
  """Calculate the deformation gradient tensor.

  The deformation gradient tensor is calculated by solving the least-squares problem
  that minimizes the sum of squared differences between the deformed coordinates and the
  coordinates obtained by applying the deformation gradient to the reference coordinates.

  Args:
      reference_coordinates (jnp.ndarray): Reference coordinates (N, 3).
      deformed_coordinates (jnp.ndarray): Deformed coordinates (N, 3).

  Returns:
      jnp.ndarray: Deformation gradient tensor (3, 3).

  """
  reference_gram_matrix = reference_coordinates.T @ reference_coordinates
  reference_gram_inv = jnp.linalg.inv(reference_gram_matrix)
  return (deformed_coordinates.T @ reference_coordinates) @ reference_gram_inv


@jit
def calculate_green_lagrange_strain(deformation_gradient: jnp.ndarray) -> jnp.ndarray:
  """Calculate the Green-Lagrange strain tensor.

  Args:
      deformation_gradient (jnp.ndarray): Deformation gradient tensor F (3, 3).

  Returns:
      jnp.ndarray: Green-Lagrange strain tensor (3, 3).

  """
  right_cauchy_green = deformation_gradient.T @ deformation_gradient
  identity_matrix = jnp.eye(3)
  return 0.5 * (right_cauchy_green - identity_matrix)


@jit
def calculate_principal_strains(green_lagrange_strain: jnp.ndarray) -> jnp.ndarray:
  """Calculate the principal strains.

  The principal strains are the eigenvalues of the Green-Lagrange strain tensor E.

  Args:
      green_lagrange_strain (jnp.ndarray): Green-Lagrange strain tensor (3, 3).

  Returns:
      jnp.ndarray: Principal strains (3,).

  """
  return jnp.linalg.eigvalsh(green_lagrange_strain)


def run_psa(reference_structure: ProteinStructure, deformed_structure: ProteinStructure) -> dict:
  """Run the full Protein Strain Analysis.

  Args:
      reference_structure (ProteinStructure): The reference protein structure.
      deformed_structure (ProteinStructure): The deformed protein structure.

  Returns:
      dict: A dictionary containing the deformation gradient, Green-Lagrange strain, and principal strains.

  """
  reference_coordinates = reference_structure.coordinates
  deformed_coordinates = deformed_structure.coordinates

  deformation_gradient = calculate_deformation_gradient(reference_coordinates, deformed_coordinates)
  green_lagrange_strain = calculate_green_lagrange_strain(deformation_gradient)
  principal_strains = calculate_principal_strains(green_lagrange_strain)

  return {
    "deformation_gradient": deformation_gradient,
    "green_lagrange_strain": green_lagrange_strain,
    "principal_strains": principal_strains,
  }
