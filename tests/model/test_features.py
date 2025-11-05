"""Tests for the features module."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.model.features import (
    embed_edges,
    encode_positions,
    extract_features,
    get_edge_chains_neighbors,
    project_features,
)


@pytest.fixture
def model_parameters():
    """Create a dummy set of model parameters for testing."""
    key = jax.random.PRNGKey(0)
    params = {
        "protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear": {
            "w": jax.random.normal(key, (66, 16)),
            "b": jax.random.normal(key, (16,)),
        },
        "protein_mpnn/~/protein_features/~/edge_embedding": {
            "w": jax.random.normal(key, (416, 128)),
            "b": jax.random.normal(key, (128,)),
        },
        "protein_mpnn/~/protein_features/~/norm_edges": {
            "scale": jnp.ones(128),
            "offset": jnp.zeros(128),
        },
        "protein_mpnn/~/W_e": {
            "w": jax.random.normal(key, (128, 6)),
            "b": jax.random.normal(key, (6,)),
        },
    }
    return params


@pytest.fixture
def features_data():
    """Create dummy data for the features module."""
    key = jax.random.PRNGKey(0)
    structure_coordinates = jnp.array(np.random.rand(10, 4, 3))
    mask = jnp.ones(10)
    residue_index = jnp.arange(10)
    chain_index = jnp.zeros(10, dtype=jnp.int32)
    neighbor_indices = jnp.array(np.random.randint(0, 10, (10, 5)))
    return (
        key,
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        neighbor_indices,
    )


def test_get_edge_chains_neighbors(features_data):
    """Test the get_edge_chains_neighbors function."""
    _, _, _, _, chain_index, neighbor_indices = features_data
    edge_chains = get_edge_chains_neighbors(chain_index, neighbor_indices)
    chex.assert_shape(edge_chains, (10, 5))


def test_encode_positions(model_parameters):
    """Test the encode_positions function."""
    neighbor_offsets = jnp.ones((10, 5), dtype=jnp.int32)
    edge_chains_neighbors = jnp.ones((10, 5), dtype=jnp.int32)
    encoded_positions = encode_positions(
        neighbor_offsets, edge_chains_neighbors, model_parameters
    )
    chex.assert_shape(encoded_positions, (10, 5, 16))


def test_embed_edges(model_parameters):
    """Test the embed_edges function."""
    edge_features = jnp.ones((10, 5, 416))
    embedded_edges = embed_edges(edge_features, model_parameters)
    chex.assert_shape(embedded_edges, (10, 5, 128))


@pytest.mark.parametrize("backbone_noise", [None, 0.1])
def test_extract_features(model_parameters, features_data, backbone_noise):
    """Test the extract_features function."""
    (
        key,
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        _,
    ) = features_data
    edge_features, neighbor_indices, _ = extract_features(
        key,
        model_parameters,
        structure_coordinates,
        mask,
        residue_index,
        chain_index,
        backbone_noise=backbone_noise,
    )
    chex.assert_shape(edge_features, (10, 10, 128))
    chex.assert_shape(neighbor_indices, (10, 10))


def test_project_features(model_parameters):
    """Test the project_features function."""
    edge_features = jnp.ones((10, 5, 128))
    projected_features = project_features(model_parameters, edge_features)
    chex.assert_shape(projected_features, (10, 5, 6))
