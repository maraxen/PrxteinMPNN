"""Utility functions for manipulating atomic coordinates."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.utils.types import (
  AtomicCoordinate,
  BackboneCoordinates,
  StructureAtomicCoordinates,
)


@partial(jax.jit, static_argnames=("augment_eps",))
def apply_noise_to_coordinates(
  coordinates: StructureAtomicCoordinates,
  key: PRNGKeyArray,
  augment_eps: float = 0.0,
) -> StructureAtomicCoordinates:
  """Add Gaussian noise to atomic coordinates."""
  if augment_eps > 0:
    noise = augment_eps * jax.random.normal(key, coordinates.shape)
    return coordinates + noise
  return coordinates


@jax.jit
def compute_backbone_coordinates(
  coordinates: StructureAtomicCoordinates,
) -> BackboneCoordinates:
  """Compute backbone coordinates from atomic coordinates."""
  nitrogen, alpha_carbon, carbon, oxygen = (
    coordinates[:, 0, :],
    coordinates[:, 1, :],
    coordinates[:, 2, :],
    coordinates[:, 3, :],
  )
  alpha_to_nitrogen = alpha_carbon - nitrogen
  carbon_to_alpha = carbon - alpha_carbon
  c_beta = compute_c_beta(
    alpha_to_nitrogen,
    carbon_to_alpha,
    alpha_carbon,
  )
  return jnp.stack(
    [nitrogen, alpha_carbon, carbon, oxygen, c_beta],
    axis=1,
  )


@jax.jit
def compute_c_beta(
  alpha_to_nitrogen: AtomicCoordinate,
  carbon_to_alpha: AtomicCoordinate,
  alpha_carbon: AtomicCoordinate,
) -> AtomicCoordinate:
  """Compute inferred C-beta coordinates from bond vectors.

  Uses a linear combination of the bond vectors to estimate C-beta coordinates.

  Coefficients are derived from empirical data and are used to ensure
  that the C-beta coordinates are consistent with the geometry of the protein backbone.

  Args:
    alpha_to_nitrogen: Bond vector from nitrogen to alpha carbon.
    carbon_to_alpha: Bond vector from alpha carbon to carbon.
    alpha_carbon: Coordinates of the alpha carbon atom.

  Returns:
    C-beta coordinates as an AtomicCoordinate.

  """
  f1, f2, f3 = -0.58273431, 0.56802827, -0.54067466
  term1 = f1 * jnp.cross(alpha_to_nitrogen, carbon_to_alpha)
  term2 = f2 * alpha_to_nitrogen
  term3 = f3 * carbon_to_alpha
  return term1 + term2 + term3 + alpha_carbon


@jax.jit
def compute_backbone_distance(backbone_coordinates: BackboneCoordinates) -> BackboneCoordinates:
  """Compute pairwise distances between backbone atoms.

  Calculate the Euclidean distance between all pairs of backbone atom coordinates based on alpha
  carbon positions.

  Assumes backbone_coordinates is a 3D array of shape (N, 5, 3), where N is the number of atoms,
  5 is the number of backbone atoms (N, CA, C, O, N), and 3 is the spatial dimension (x, y, z).

  Args:
    backbone_coordinates: A 3D array of shape (N, 5, 3) representing the coordinates of backbone
    atoms.

  Returns:
    A 2D array of shape (N, N) containing the pairwise distances between backbone atoms.

  """
  return jnp.sqrt(
    1e-6
    + jnp.sum(
      (backbone_coordinates[:, None, :, :] - backbone_coordinates[None, :, :, :]) ** 2,
      axis=-1,
    ),
  )
