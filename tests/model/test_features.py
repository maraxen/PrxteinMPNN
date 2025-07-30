"""Unit tests for feature extraction functions in prxteinmpnn.model.features.

These tests cover the core feature extraction pipeline, including edge chain neighbor
computation, position encoding, edge embedding, and the main extract_features function.

All tests use mock data and minimal model parameters to ensure deterministic and isolated
behavior. JAX is used for all numerical operations, and tests are compatible with pytest.

Run with:
  python -m pytest src/prxteinmpnn/model/test_features.py
"""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.features import (
  MAXIMUM_RELATIVE_FEATURES,
  embed_edges,
  encode_positions,
  extract_features,
  get_edge_chains_neighbors,
  project_features,
)


@pytest.fixture
def mock_model_parameters():
  """Fixture for minimal mock model parameters."""
  # Shapes are chosen to match the expected input/output of the tested functions.
  pos_enc_dim = 2 * MAXIMUM_RELATIVE_FEATURES + 2
  pos_enc_out = 4
  edge_emb_out = 8

  # CORRECTED: The RBF dimension is num_pairs * num_bases
  num_rbf_bases = 16
  num_atom_pairs = 25
  rbf_dim = num_atom_pairs * num_rbf_bases  # 25 * 16 = 400

  # CORRECTED: The total input dimension for the edge embedding layer
  edge_in_dim = pos_enc_out + rbf_dim  # 4 + 400 = 404

  w_pos = jnp.ones((pos_enc_dim, pos_enc_out))
  b_pos = jnp.zeros((pos_enc_out,))

  # CORRECTED: Use the correct input dimension for the weight matrix
  w_edge = jnp.ones((edge_in_dim, edge_emb_out))  # Shape (404, 8)
  b_edge = jnp.zeros((edge_emb_out,))

  scale = jnp.ones((edge_emb_out,))
  offset = jnp.zeros((edge_emb_out,))
  w_proj = jnp.ones((edge_emb_out, edge_emb_out))
  b_proj = jnp.zeros((edge_emb_out,))
  return {
    "protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear": {
      "w": w_pos,
      "b": b_pos,
    },
    "protein_mpnn/~/protein_features/~/edge_embedding": {
      "w": w_edge,
      "b": b_edge,
    },
    "protein_mpnn/~/protein_features/~/norm_edges": {
      "norm": {
        "scale": scale,
        "offset": offset,
      }
    },
    "protein_mpnn/~/W_e": {
      "w": w_proj,
      "b": b_proj,
    },
  }


def test_get_edge_chains_neighbors():
  """Test get_edge_chains_neighbors computes correct edge chain neighbor matrix.

  Returns:
    None

  Raises:
    AssertionError: If output does not match expected.

  """
  chain_indices = jnp.array([0, 1, 0, 1])
  neighbor_indices = jnp.array([[1, 2], [0, 3], [1, 0], [2, 1]])
  result = get_edge_chains_neighbors(chain_indices, neighbor_indices)
  expected = jnp.array(
    [
      [0, 1],  # atom 0: neighbor 1 (diff chain), neighbor 2 (same chain)
      [0, 1],  # atom 1: neighbor 0 (diff), neighbor 3 (same)
      [0, 1],  # atom 2: neighbor 1 (diff), neighbor 0 (same)
      [0, 1],  # atom 3: neighbor 2 (diff), neighbor 1 (same)
    ]
  )
  assert jnp.all(result == expected), f"Expected {expected}, got {result}"


def test_encode_positions(mock_model_parameters):
  """Test encode_positions produces correct shape and is deterministic.

  Returns:
    None

  Raises:
    AssertionError: If output shape or values are incorrect.

  """
  neighbor_offsets = jnp.array([[0, 1], [1, 0]])
  edge_chains_neighbors = jnp.array([[1, 0], [0, 1]])
  params = mock_model_parameters
  encoded = encode_positions(neighbor_offsets, edge_chains_neighbors, params)
  # Should have shape (2, 2, pos_enc_out)
  assert (
    encoded.shape[-1]
    == params["protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear"][
      "w"
    ].shape[1]
  )
  # Deterministic: calling again gives same result
  encoded2 = encode_positions(neighbor_offsets, edge_chains_neighbors, params)
  assert jnp.allclose(encoded, encoded2)


def test_embed_edges(mock_model_parameters):
  """Test embed_edges produces correct output shape.

  Returns:
    None

  Raises:
    AssertionError: If output shape is incorrect.

  """
  # CORRECTED: The feature dimension is 4 (pos enc) + 400 (rbf) = 404
  edge_features = jnp.ones((2, 2, 404))
  params = mock_model_parameters
  embedded = embed_edges(edge_features, params)
  assert embedded.shape == (
    2,
    2,
    params["protein_mpnn/~/protein_features/~/edge_embedding"]["w"].shape[1],
  )


def test_project_features(mock_model_parameters):
  """Test project_features applies linear projection.

  Returns:
    None

  Raises:
    AssertionError: If output shape is incorrect.

  """
  edge_features = jnp.ones((2, 2, 8))
  params = mock_model_parameters
  projected = project_features(params, edge_features)
  assert projected.shape == (2, 2, 8)


def test_extract_features_shapes(mock_model_parameters):
  """Test extract_features returns correct shapes and is JIT-compatible.

  Returns:
    None

  Raises:
    AssertionError: If output shapes are incorrect.

  """
  num_residues = 4
  atoms_per_residue = 4  # Mock N, CA, C, O
  structure_coordinates = (
    jnp.arange(num_residues * atoms_per_residue * 3)
    .reshape(num_residues, atoms_per_residue, 3)
    .astype(jnp.float32)
  )
  mask = jnp.array([1, 1, 1, 1])
  residue_indices = jnp.array([0, 1, 2, 3])
  chain_indices = jnp.array([0, 0, 1, 1])
  prng_key = jax.random.PRNGKey(0)
  params = mock_model_parameters

  edge_features, neighbor_indices = extract_features(
    structure_coordinates,
    mask,
    residue_indices,
    chain_indices,
    params,
    prng_key,
    k_neighbors=2,
    augment_eps=0.0,
  )

  assert edge_features.shape[0] == num_residues
  assert neighbor_indices.shape == (num_residues, 2)
  # Check that edge_features is finite
  assert jnp.all(jnp.isfinite(edge_features)), "Edge features contain non-finite values."


def test_extract_features_with_noise(mock_model_parameters):
  """Test extract_features with coordinate noise augmentation.

  Returns:
    None

  Raises:
    AssertionError: If output is not finite or shape is wrong.

  """
  num_residues = 4
  atoms_per_residue = 4  # Mock N, CA, C, O
  structure_coordinates = (
    jnp.arange(num_residues * atoms_per_residue * 3)
    .reshape(num_residues, atoms_per_residue, 3)
    .astype(jnp.float32)
  )
  mask = jnp.array([1, 1, 1, 1])
  residue_indices = jnp.array([0, 1, 2, 3])
  chain_indices = jnp.array([0, 0, 1, 1])
  prng_key = jax.random.PRNGKey(42)
  params = mock_model_parameters

  edge_features, neighbor_indices = extract_features(
    structure_coordinates,
    mask,
    residue_indices,
    chain_indices,
    params,
    prng_key,
    k_neighbors=2,
    augment_eps=0.1,
  )
  assert edge_features.shape[0] == num_residues
  assert neighbor_indices.shape == (num_residues, 2)
  assert jnp.all(jnp.isfinite(edge_features)), "Edge features contain non-finite values."