"""Tests for structural metrics and distance calculations."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from prxteinmpnn.utils.structure_metrics import (
  calculate_ca_distance_matrix,
  calculate_cb_distance_matrix,
  calculate_closest_atom_distance_matrix,
  calculate_cosine_similarity,
  calculate_rmsd,
  calculate_tm_score,
)


@pytest.fixture
def sample_ca_coordinates():
  """Create sample C-alpha coordinates for testing.

  Returns:
    JAX array of shape (5, 3) representing 5 C-alpha atoms.

  """
  return jnp.array([
    [0.0, 0.0, 0.0],
    [3.8, 0.0, 0.0],
    [7.6, 0.0, 0.0],
    [11.4, 0.0, 0.0],
    [15.2, 0.0, 0.0],
  ])


@pytest.fixture
def sample_cb_coordinates():
  """Create sample C-beta coordinates for testing.

  Returns:
    JAX array of shape (5, 3) representing 5 C-beta atoms.

  """
  return jnp.array([
    [0.0, 1.5, 0.0],
    [3.8, 1.5, 0.0],
    [7.6, 1.5, 0.0],
    [11.4, 1.5, 0.0],
    [15.2, 1.5, 0.0],
  ])


@pytest.fixture
def sample_full_coordinates():
  """Create sample full atom coordinates for testing.

  Returns:
    JAX array of shape (5, 37, 3) representing 5 residues with up to 37 atoms.

  """
  return jnp.ones((5, 37, 3))


@pytest.fixture
def sample_atom_mask():
  """Create sample atom mask for testing.

  Returns:
    JAX array of shape (5, 37) indicating present atoms.

  """
  return jnp.ones((5, 37))


def test_calculate_ca_distance_matrix_not_implemented(sample_ca_coordinates):
  """Test that calculate_ca_distance_matrix raises NotImplementedError.

  Args:
    sample_ca_coordinates: Fixture providing sample C-alpha coordinates.

  Raises:
    AssertionError: If NotImplementedError is not raised.

  """
  with pytest.raises(NotImplementedError):
    calculate_ca_distance_matrix(sample_ca_coordinates)


def test_calculate_cb_distance_matrix_not_implemented(sample_cb_coordinates):
  """Test that calculate_cb_distance_matrix raises NotImplementedError.

  Args:
    sample_cb_coordinates: Fixture providing sample C-beta coordinates.

  Raises:
    AssertionError: If NotImplementedError is not raised.

  """
  with pytest.raises(NotImplementedError):
    calculate_cb_distance_matrix(sample_cb_coordinates)


def test_calculate_closest_atom_distance_matrix_not_implemented(
  sample_full_coordinates,
  sample_atom_mask,
):
  """Test that calculate_closest_atom_distance_matrix raises NotImplementedError.

  Args:
    sample_full_coordinates: Fixture providing full atomic coordinates.
    sample_atom_mask: Fixture providing atom mask.

  Raises:
    AssertionError: If NotImplementedError is not raised.

  """
  with pytest.raises(NotImplementedError):
    calculate_closest_atom_distance_matrix(sample_full_coordinates, sample_atom_mask)


def test_calculate_rmsd_not_implemented(sample_ca_coordinates):
  """Test that calculate_rmsd raises NotImplementedError.

  Args:
    sample_ca_coordinates: Fixture providing sample C-alpha coordinates.

  Raises:
    AssertionError: If NotImplementedError is not raised.

  """
  coords1 = sample_ca_coordinates
  coords2 = sample_ca_coordinates + 0.1
  with pytest.raises(NotImplementedError):
    calculate_rmsd(coords1, coords2)


def test_calculate_rmsd_with_alignment_not_implemented(sample_ca_coordinates):
  """Test calculate_rmsd with align=True raises NotImplementedError.

  Args:
    sample_ca_coordinates: Fixture providing sample C-alpha coordinates.

  Raises:
    AssertionError: If NotImplementedError is not raised.

  """
  coords1 = sample_ca_coordinates
  coords2 = sample_ca_coordinates + 0.1
  with pytest.raises(NotImplementedError):
    calculate_rmsd(coords1, coords2, align=True)


def test_calculate_tm_score_not_implemented(sample_ca_coordinates):
  """Test that calculate_tm_score raises NotImplementedError.

  Args:
    sample_ca_coordinates: Fixture providing sample C-alpha coordinates.

  Raises:
    AssertionError: If NotImplementedError is not raised.

  """
  coords1 = sample_ca_coordinates
  coords2 = sample_ca_coordinates + 0.1
  with pytest.raises(NotImplementedError):
    calculate_tm_score(coords1, coords2, sequence_length=5)


def test_calculate_cosine_similarity_not_implemented():
  """Test that calculate_cosine_similarity raises NotImplementedError.

  Raises:
    AssertionError: If NotImplementedError is not raised.

  """
  feat1 = jnp.array([1.0, 0.0, 0.0])
  feat2 = jnp.array([1.0, 0.0, 0.0])
  with pytest.raises(NotImplementedError):
    calculate_cosine_similarity(feat1, feat2)
