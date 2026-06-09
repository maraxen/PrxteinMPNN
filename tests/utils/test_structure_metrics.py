"""Tests for structural metrics and distance calculations."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from aminx.utils.structure_metrics import (
  calculate_ca_distance_matrix,
  calculate_cb_distance_matrix,
  calculate_closest_atom_distance_matrix,
  calculate_cosine_similarity,
  calculate_rmsd,
  calculate_tm_score,
)


@pytest.fixture
def sample_ca_coordinates():
  """Create sample C-alpha coordinates for testing."""
  return jnp.array([
    [0.0, 0.0, 0.0],
    [3.8, 0.0, 0.0],
    [7.6, 0.0, 0.0],
    [11.4, 0.0, 0.0],
    [15.2, 0.0, 0.0],
  ])


@pytest.fixture
def sample_cb_coordinates():
  """Create sample C-beta coordinates for testing."""
  return jnp.array([
    [0.0, 1.5, 0.0],
    [3.8, 1.5, 0.0],
    [7.6, 1.5, 0.0],
    [11.4, 1.5, 0.0],
    [15.2, 1.5, 0.0],
  ])


@pytest.fixture
def sample_full_coordinates():
  """Create sample full atom coordinates for testing."""
  coords = jnp.zeros((5, 37, 3))
  coords = coords.at[:, 1, 0].set(jnp.arange(5) * 3.8)
  return coords


@pytest.fixture
def sample_atom_mask():
  """Create sample atom mask for testing."""
  return jnp.ones((5, 37))


def test_calculate_ca_distance_matrix(sample_ca_coordinates):
  dist = calculate_ca_distance_matrix(sample_ca_coordinates)
  assert dist.shape == (5, 5)
  assert pytest.approx(float(dist[0, 1]), rel=1e-5) == 3.8
  assert float(dist[0, 0]) == 0.0


def test_calculate_cb_distance_matrix(sample_cb_coordinates):
  dist = calculate_cb_distance_matrix(sample_cb_coordinates)
  assert dist.shape == (5, 5)
  assert pytest.approx(float(dist[0, 1]), rel=1e-5) == 3.8


def test_calculate_closest_atom_distance_matrix(
  sample_full_coordinates,
  sample_atom_mask,
):
  dist = calculate_closest_atom_distance_matrix(sample_full_coordinates, sample_atom_mask)
  assert dist.shape == (5, 5)
  assert float(dist[0, 0]) == 0.0


def test_calculate_rmsd(sample_ca_coordinates):
  coords1 = sample_ca_coordinates
  coords2 = sample_ca_coordinates + 0.1
  rmsd = float(calculate_rmsd(coords1, coords2, align=False))
  assert rmsd == pytest.approx(0.1, abs=1e-5)


def test_calculate_rmsd_with_alignment(sample_ca_coordinates):
  coords1 = sample_ca_coordinates
  coords2 = sample_ca_coordinates + 0.1
  rmsd = float(calculate_rmsd(coords1, coords2, align=True))
  assert rmsd < 0.2


def test_calculate_tm_score(sample_ca_coordinates):
  coords1 = sample_ca_coordinates
  coords2 = sample_ca_coordinates + 0.1
  tm = float(calculate_tm_score(coords1, coords2, sequence_length=5))
  assert 0.0 <= tm <= 1.0


def test_calculate_cosine_similarity():
  feat1 = jnp.array([1.0, 0.0, 0.0])
  feat2 = jnp.array([1.0, 0.0, 0.0])
  assert float(calculate_cosine_similarity(feat1, feat2)) == pytest.approx(1.0)
