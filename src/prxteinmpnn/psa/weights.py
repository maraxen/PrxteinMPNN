"""JAX-native functions for calculating neighborhood weights for Protein Strain Analysis."""

import jax.numpy as jnp


def linear_weights(
  coordinates: jnp.ndarray,
  r_inner: float,
  r_outer: float,
) -> jnp.ndarray:
  """Calculate weights for each pair of atoms using a linear decay function.

  The weight is 1 if the distance is less than r_inner, 0 if it's greater than r_outer,
  and decays linearly from 1 to 0 between r_inner and r_outer.

  Args:
      coordinates (jnp.ndarray): The coordinates of the atoms (N, 3).
      r_inner (float): The inner radius for the weighting function.
      r_outer (float): The outer radius for the weighting function.

  Returns:
      jnp.ndarray: A weight matrix (N, N).

  """
  # Calculate pairwise distances
  diffs = coordinates[:, None, :] - coordinates[None, :, :]
  distances = jnp.linalg.norm(diffs, axis=-1)

  # Linear decay function
  weights = (distances - r_inner) / (r_outer - r_inner)
  weights = 1.0 - weights

  # Cap weights between 0 and 1
  return jnp.maximum(0.0, jnp.minimum(1.0, weights))
