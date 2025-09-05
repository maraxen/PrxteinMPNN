"""JAX-native functions for spatial transformations."""

import jax.numpy as jnp


def cartesian_to_cylindrical(coordinates: jnp.ndarray) -> jnp.ndarray:
  """Transform coordinates from Cartesian to cylindrical.

  Args:
      coordinates (jnp.ndarray): Cartesian coordinates (N, 3).

  Returns:
      jnp.ndarray: Cylindrical coordinates (N, 3) in (r, theta, z) format.

  """
  x, y, z = coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
  r = jnp.sqrt(x**2 + y**2)
  theta = jnp.arctan2(y, x)
  return jnp.stack([r, theta, z], axis=-1)
