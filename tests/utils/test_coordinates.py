"""Tests for coordinate utilities."""

import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.utils.coordinates import (
  apply_noise_to_coordinates,
  compute_backbone_coordinates,
  compute_backbone_distance,
  compute_c_beta,
  compute_cb_precise,
  extend_coordinate,
)

KEY = jax.random.PRNGKey(42)


def test_apply_noise_to_coordinates():
  """Test adding Gaussian noise to atomic coordinates.

  Raises:
      AssertionError: If the output does not match the expected value.
  """
  coords = jnp.ones((10, 5, 3))

  # Test with no noise
  coords_no_noise = apply_noise_to_coordinates(coords, KEY, augment_eps=0.0)
  chex.assert_trees_all_equal(coords, coords_no_noise)

  # Test with noise
  coords_with_noise = apply_noise_to_coordinates(coords, KEY, augment_eps=0.1)
  chex.assert_shape(coords, coords_with_noise.shape)
  assert not jnp.allclose(coords, coords_with_noise)


def test_compute_c_beta():
  """Test the computation of C-beta coordinates.

  Raises:
      AssertionError: If the output does not match the expected value.
  """
  alpha_carbon = jnp.array([0.0, 0.0, 0.0])
  alpha_to_nitrogen = jnp.array([1.0, 0.0, 0.0])
  carbon_to_alpha = jnp.array([0.0, 1.0, 0.0])

  # Manually computed expected result based on the function's formula
  f1, f2, f3 = -0.58273431, 0.56802827, -0.54067466
  term1 = f1 * jnp.cross(alpha_to_nitrogen, carbon_to_alpha)
  term2 = f2 * alpha_to_nitrogen
  term3 = f3 * carbon_to_alpha
  expected_cb = term1 + term2 + term3 + alpha_carbon

  computed_cb = compute_c_beta(alpha_to_nitrogen, carbon_to_alpha, alpha_carbon)
  chex.assert_trees_all_close(computed_cb, expected_cb)


def test_compute_backbone_coordinates():
  """Test the computation of backbone coordinates.

  Raises:
      AssertionError: If the output does not match the expected value.
  """
  coords = jnp.arange(10 * 5 * 3).reshape((10, 5, 3)).astype(jnp.float32)
  backbone_coords = compute_backbone_coordinates(coords)

  chex.assert_shape(backbone_coords, (10, 5, 3))
  # N, CA, C, O should be passed through
  chex.assert_trees_all_equal(coords[:, :4, :], backbone_coords[:, :4, :])
  # CB should be different
  assert not jnp.allclose(coords[:, 4, :], backbone_coords[:, 4, :])


def test_compute_backbone_distance():
  """Test computation of pairwise distances between backbone alpha carbons.

  Raises:
      AssertionError: If the output does not match the expected value.
  """
  coords = jnp.zeros((2, 5, 3), dtype=jnp.float32)
  coords = coords.at[1, 1, 0].set(3.0)  # Distance of 3 between the two CA atoms

  distances = compute_backbone_distance(coords)
  # The shape should be (N, N) as it's the distance between corresponding atoms
  chex.assert_shape(distances, (2, 2))
  chex.assert_trees_all_close(distances[0, 0], jnp.sqrt(1e-6))
  chex.assert_trees_all_close(distances[0, 1], 3.0, atol=1e-3)


def test_extend_coordinate():
  """Test the extension of coordinates to a fourth atom.

  Raises:
      AssertionError: If the output does not match the expected value.
  """
  a = jnp.array([1.0, 0.0, 0.0])
  b = jnp.array([0.0, 0.0, 0.0])
  c = jnp.array([0.0, 1.0, 0.0])

  # Place D in the xy-plane
  d = extend_coordinate(a, b, c, bond_length=1.0, bond_angle=jnp.pi / 2, dihedral_angle=0.0)
  # Corrected expected value
  chex.assert_trees_all_close(d, jnp.array([1.0, 1.0, 0.0]), atol=1e-6)

  # Place D out of the xy-plane
  d_dihedral = extend_coordinate(
    a, b, c, bond_length=1.0, bond_angle=jnp.pi / 2, dihedral_angle=jnp.pi / 2
  )
  chex.assert_trees_all_close(d_dihedral, jnp.array([0.0, 1.0, -1.0]), atol=1e-6)


def test_compute_cb_precise():
  """Test the precise computation of C-beta coordinates.

  Raises:
      AssertionError: If the output does not match the expected value.
  """
  n = jnp.array([1.45, 0.0, 0.0])
  ca = jnp.array([0.0, 0.0, 0.0])
  c = jnp.array([0.0, 1.53, 0.0])

  cb = compute_cb_precise(n, ca, c)
  chex.assert_shape(cb, (3,))

  # Check if bond length from CA to CB is roughly correct
  bond_length = jnp.linalg.norm(cb - ca)
  chex.assert_trees_all_close(bond_length, 1.522, atol=1e-3)