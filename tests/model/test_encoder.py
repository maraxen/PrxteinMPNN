"""Tests for the encoder module."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.model.encoder import (
    encode,
    encoder_normalize,
    encoder_parameter_pytree,
    initialize_node_features,
    make_encoder,
)


@pytest.fixture
def model_parameters():
    """Create a dummy set of model parameters for testing."""
    key = jax.random.PRNGKey(0)
    params = {}
    for i in range(3):
        prefix = "protein_mpnn/~/enc_layer"
        if i > 0:
            prefix += f"_{i}"
        layer_name_suffix = f"enc{i}"
        params[f"{prefix}/~/{layer_name_suffix}_W1"] = {
            "w": jax.random.normal(key, (262, 128)),
            "b": jax.random.normal(key, (128,)),
        }
        params[f"{prefix}/~/{layer_name_suffix}_W2"] = {
            "w": jax.random.normal(key, (128, 128)),
            "b": jax.random.normal(key, (128,)),
        }
        params[f"{prefix}/~/{layer_name_suffix}_W3"] = {
            "w": jax.random.normal(key, (128, 128)),
            "b": jax.random.normal(key, (128,)),
        }
        params[f"{prefix}/~/{layer_name_suffix}_norm1"] = {
            "scale": jnp.ones(128),
            "offset": jnp.zeros(128),
        }
        params[
            f"{prefix}/~/position_wise_feed_forward/~/{layer_name_suffix}_dense_W_in"
        ] = {"w": jax.random.normal(key, (128, 512)), "b": jax.random.normal(key, (512,))}
        params[
            f"{prefix}/~/position_wise_feed_forward/~/{layer_name_suffix}_dense_W_out"
        ] = {"w": jax.random.normal(key, (512, 128)), "b": jax.random.normal(key, (128,))}
        params[f"{prefix}/~/{layer_name_suffix}_norm2"] = {
            "scale": jnp.ones(128),
            "offset": jnp.zeros(128),
        }
        params[f"{prefix}/~/{layer_name_suffix}_W11"] = {
            "w": jax.random.normal(key, (262, 128)),
            "b": jax.random.normal(key, (128,)),
        }
        params[f"{prefix}/~/{layer_name_suffix}_W12"] = {
            "w": jax.random.normal(key, (128, 128)),
            "b": jax.random.normal(key, (128,)),
        }
        params[f"{prefix}/~/{layer_name_suffix}_W13"] = {
            "w": jax.random.normal(key, (128, 6)),
            "b": jax.random.normal(key, (6,)),
        }
        params[f"{prefix}/~/{layer_name_suffix}_norm3"] = {
            "scale": jnp.ones(6),
            "offset": jnp.zeros(6),
        }
    params["protein_mpnn/~/W_e"] = {"b": jax.random.normal(key, (128,))}
    return params


@pytest.fixture
def encoder_data():
    """Create dummy data for the encoder."""
    node_features = jnp.zeros((10, 128))
    edge_features = jnp.ones((10, 5, 6))
    neighbor_indices = jnp.array(np.random.randint(0, 10, (10, 5)))
    mask = jnp.ones(10)
    attention_mask = jnp.ones((10, 5))
    return (
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        attention_mask,
    )


def test_encoder_parameter_pytree(model_parameters):
    """Test the creation of the encoder parameter pytree."""
    pytree = encoder_parameter_pytree(model_parameters)
    chex.assert_tree_shape_prefix(pytree, (3,))


def test_initialize_node_features(model_parameters):
    """Test the initialization of node features."""
    edge_features = jnp.ones((10, 5, 6))
    node_features = initialize_node_features(model_parameters, edge_features)
    chex.assert_shape(node_features, (10, 128))


def test_encode(encoder_data):
    """Test the encode function."""
    node_features, edge_features, neighbor_indices, _, _ = encoder_data
    layer_params = {
        "W1": {"w": jnp.ones((262, 128)), "b": jnp.ones(128)},
        "W2": {"w": jnp.ones((128, 128)), "b": jnp.ones(128)},
        "W3": {"w": jnp.ones((128, 128)), "b": jnp.ones(128)},
    }
    message = encode(node_features, edge_features, neighbor_indices, layer_params)
    chex.assert_shape(message, (10, 5, 128))


def test_encoder_normalize(encoder_data):
    """Test the encoder_normalize function."""
    (
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        _,
    ) = encoder_data
    message = jnp.ones((10, 5, 128))
    layer_params = {
        "norm1": {"scale": jnp.ones(128), "offset": jnp.zeros(128)},
        "dense_W_in": {"w": jnp.ones((128, 512)), "b": jnp.ones(512)},
        "dense_W_out": {"w": jnp.ones((512, 128)), "b": jnp.ones(128)},
        "norm2": {"scale": jnp.ones(128), "offset": jnp.zeros(128)},
        "W11": {"w": jnp.ones((262, 128)), "b": jnp.ones(128)},
        "W12": {"w": jnp.ones((128, 128)), "b": jnp.ones(128)},
        "W13": {"w": jnp.ones((128, 6)), "b": jnp.ones(6)},
        "norm3": {"scale": jnp.ones(6), "offset": jnp.zeros(6)},
    }
    updated_node_features, updated_edge_features = encoder_normalize(
        message, node_features, edge_features, neighbor_indices, mask, layer_params
    )
    chex.assert_shape(updated_node_features, (10, 128))
    chex.assert_shape(updated_edge_features, (10, 5, 6))


@pytest.mark.parametrize("attention_mask_type", [None, "seq_mask"])
def test_make_encoder(model_parameters, encoder_data, attention_mask_type):
    """Test the make_encoder function."""
    _, edge_features, neighbor_indices, mask, attention_mask = encoder_data
    encoder = make_encoder(model_parameters, attention_mask_type=attention_mask_type)
    if attention_mask_type is None:
        node_features, edge_features = encoder(
            edge_features, neighbor_indices, mask
        )
    else:
        node_features, edge_features = encoder(
            edge_features, neighbor_indices, mask, attention_mask
        )
    chex.assert_shape(node_features, (10, 128))
    chex.assert_shape(edge_features, (10, 5, 6))


@pytest.mark.parametrize("attention_mask_type", [None, "seq_mask"])
def test_make_encoder_jit(model_parameters, encoder_data, attention_mask_type):
    """Test the jitted make_encoder function."""
    _, edge_features, neighbor_indices, mask, attention_mask = encoder_data
    encoder = jax.jit(
        make_encoder(model_parameters, attention_mask_type=attention_mask_type)
    )
    if attention_mask_type is None:
        node_features, edge_features = encoder(
            edge_features, neighbor_indices, mask
        )
    else:
        node_features, edge_features = encoder(
            edge_features, neighbor_indices, mask, attention_mask
        )
    chex.assert_shape(node_features, (10, 128))
    chex.assert_shape(edge_features, (10, 5, 6))


@pytest.mark.parametrize("attention_mask_type", [None, "seq_mask"])
def test_make_encoder_vmap(model_parameters, encoder_data, attention_mask_type):
    """Test the vmapped make_encoder function."""
    _, edge_features, neighbor_indices, mask, attention_mask = encoder_data
    b_edge_features = jnp.stack([edge_features] * 2)
    b_neighbor_indices = jnp.stack([neighbor_indices] * 2)
    b_mask = jnp.stack([mask] * 2)
    b_attention_mask = jnp.stack([attention_mask] * 2)

    encoder = jax.vmap(
        make_encoder(model_parameters, attention_mask_type=attention_mask_type),
        in_axes=(0, 0, 0, 0) if attention_mask_type else (0, 0, 0),
    )
    if attention_mask_type is None:
        node_features, edge_features = encoder(
            b_edge_features, b_neighbor_indices, b_mask
        )
    else:
        node_features, edge_features = encoder(
            b_edge_features, b_neighbor_indices, b_mask, b_attention_mask
        )
    chex.assert_shape(node_features, (2, 10, 128))
    chex.assert_shape(edge_features, (2, 10, 5, 6))
