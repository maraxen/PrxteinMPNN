"""Tests for the decoder module."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.model.decoder import (
    decode_message,
    decoder_normalize,
    decoder_parameter_pytree,
    embed_sequence,
    initialize_conditional_decoder,
    make_decoder,
)


@pytest.fixture
def model_parameters():
    """Create a dummy set of model parameters for testing."""
    key = jax.random.PRNGKey(0)
    params = {}
    for i in range(3):
        prefix = "protein_mpnn/~/dec_layer"
        if i > 0:
            prefix += f"_{i}"
        layer_name_suffix = f"dec{i}"
        params[f"{prefix}/~/{layer_name_suffix}_W1"] = {
            "w": jax.random.normal(key, (390, 128)),
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
    params["protein_mpnn/~/embed_token"] = {"W_s": jax.random.normal(key, (21, 128))}
    params["protein_mpnn/~/W_out"] = {
        "w": jax.random.normal(key, (128, 21)),
        "b": jax.random.normal(key, (21,)),
    }
    return params


@pytest.fixture
def decoder_data():
    """Create dummy data for the decoder."""
    key = jax.random.PRNGKey(0)
    node_features = jnp.zeros((10, 128))
    edge_features = jnp.ones((10, 5, 6))
    neighbor_indices = jnp.array(np.random.randint(0, 10, (10, 5)))
    mask = jnp.ones(10)
    attention_mask = jnp.ones((10, 5))
    one_hot_sequence = jax.nn.one_hot(jnp.zeros(10, dtype=jnp.int32), 21)
    ar_mask = jnp.ones((10, 10))
    return (
        key,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        attention_mask,
        one_hot_sequence,
        ar_mask,
    )


def test_decoder_parameter_pytree(model_parameters):
    """Test the creation of the decoder parameter pytree."""
    pytree = decoder_parameter_pytree(model_parameters)
    chex.assert_tree_shape_prefix(pytree, (3,))


def test_embed_sequence(model_parameters):
    """Test the sequence embedding."""
    one_hot_sequence = jax.nn.one_hot(jnp.zeros(10, dtype=jnp.int32), 21)
    embedded_sequence = embed_sequence(model_parameters, one_hot_sequence)
    chex.assert_shape(embedded_sequence, (10, 128))


def test_initialize_conditional_decoder(model_parameters, decoder_data):
    """Test the initialization of the conditional decoder."""
    (
        _,
        node_features,
        edge_features,
        neighbor_indices,
        _,
        _,
        one_hot_sequence,
        _,
    ) = decoder_data
    node_edge_features, sequence_edge_features = initialize_conditional_decoder(
        one_hot_sequence,
        node_features,
        edge_features,
        neighbor_indices,
        model_parameters,
    )
    chex.assert_shape(node_edge_features, (10, 5, 262))
    chex.assert_shape(sequence_edge_features, (10, 5, 134))


def test_decode_message(decoder_data):
    """Test the decode_message function."""
    _, node_features, _, _, _, _, _, _ = decoder_data
    edge_features = jnp.ones((10, 5, 262))
    layer_params = {
        "W1": {"w": jnp.ones((390, 128)), "b": jnp.ones(128)},
        "W2": {"w": jnp.ones((128, 128)), "b": jnp.ones(128)},
        "W3": {"w": jnp.ones((128, 128)), "b": jnp.ones(128)},
    }
    message = decode_message(node_features, edge_features, layer_params)
    chex.assert_shape(message, (10, 5, 128))


def test_decoder_normalize(decoder_data):
    """Test the decoder_normalize function."""
    _, node_features, _, _, mask, _, _, _ = decoder_data
    message = jnp.ones((10, 5, 128))
    layer_params = {
        "norm1": {"scale": jnp.ones(128), "offset": jnp.zeros(128)},
        "dense_W_in": {"w": jnp.ones((128, 512)), "b": jnp.ones(512)},
        "dense_W_out": {"w": jnp.ones((512, 128)), "b": jnp.ones(128)},
        "norm2": {"scale": jnp.ones(128), "offset": jnp.zeros(128)},
    }
    node_features = decoder_normalize(message, node_features, mask, layer_params)
    chex.assert_shape(node_features, (10, 128))


@pytest.mark.parametrize(
    "decoding_approach, attention_mask_type",
    [
        ("unconditional", None),
        ("unconditional", "seq_mask"),
        ("conditional", "conditional"),
        ("autoregressive", None),
    ],
)
def test_make_decoder(
    model_parameters, decoder_data, decoding_approach, attention_mask_type
):
    """Test the make_decoder function for all decoding approaches."""
    (
        key,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        attention_mask,
        one_hot_sequence,
        ar_mask,
    ) = decoder_data
    decoder = make_decoder(
        model_parameters,
        attention_mask_type=attention_mask_type,
        decoding_approach=decoding_approach,
    )

    if decoding_approach == "unconditional":
        if attention_mask_type is None:
            output = decoder(node_features, edge_features, mask)
        else:
            nodes_expanded = jnp.tile(
                jnp.expand_dims(node_features, -2),
                [1, edge_features.shape[1], 1],
            )
            zeros_expanded = jnp.tile(
                jnp.expand_dims(jnp.zeros_like(node_features), -2),
                [1, edge_features.shape[1], 1],
            )
            decoder_input_features = jnp.concatenate(
                [nodes_expanded, zeros_expanded, edge_features],
                -1,
            )
            output = decoder(
                node_features, decoder_input_features, mask, attention_mask
            )
        chex.assert_shape(output, (10, 128))

    elif decoding_approach == "conditional":
        output = decoder(
            node_features,
            edge_features,
            neighbor_indices,
            mask,
            ar_mask,
            one_hot_sequence,
        )
        chex.assert_shape(output, (10, 128))

    elif decoding_approach == "autoregressive":
        sequence, logits = decoder(
            key, node_features, edge_features, neighbor_indices, mask, ar_mask
        )
        chex.assert_shape(sequence, (10, 21))
        chex.assert_shape(logits, (10, 21))


@pytest.mark.parametrize(
    "decoding_approach, attention_mask_type",
    [
        ("unconditional", None),
        ("unconditional", "seq_mask"),
        ("conditional", "conditional"),
        ("autoregressive", None),
    ],
)
def test_make_decoder_jit(
    model_parameters, decoder_data, decoding_approach, attention_mask_type
):
    """Test the jitted make_decoder function."""
    (
        key,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        attention_mask,
        one_hot_sequence,
        ar_mask,
    ) = decoder_data
    decoder = jax.jit(
        make_decoder(
            model_parameters,
            attention_mask_type=attention_mask_type,
            decoding_approach=decoding_approach,
        )
    )

    if decoding_approach == "unconditional":
        if attention_mask_type is None:
            output = decoder(node_features, edge_features, mask)
        else:
            nodes_expanded = jnp.tile(
                jnp.expand_dims(node_features, -2),
                [1, edge_features.shape[1], 1],
            )
            zeros_expanded = jnp.tile(
                jnp.expand_dims(jnp.zeros_like(node_features), -2),
                [1, edge_features.shape[1], 1],
            )
            decoder_input_features = jnp.concatenate(
                [nodes_expanded, zeros_expanded, edge_features],
                -1,
            )
            output = decoder(
                node_features, decoder_input_features, mask, attention_mask
            )
        chex.assert_shape(output, (10, 128))

    elif decoding_approach == "conditional":
        output = decoder(
            node_features,
            edge_features,
            neighbor_indices,
            mask,
            ar_mask,
            one_hot_sequence,
        )
        chex.assert_shape(output, (10, 128))

    elif decoding_approach == "autoregressive":
        sequence, logits = decoder(
            key, node_features, edge_features, neighbor_indices, mask, ar_mask
        )
        chex.assert_shape(sequence, (10, 21))
        chex.assert_shape(logits, (10, 21))


@pytest.mark.parametrize(
    "decoding_approach, attention_mask_type",
    [
        ("unconditional", None),
        ("unconditional", "seq_mask"),
        ("conditional", "conditional"),
    ],
)
def test_make_decoder_vmap(
    model_parameters, decoder_data, decoding_approach, attention_mask_type
):
    """Test the vmapped make_decoder function."""
    (
        key,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        attention_mask,
        one_hot_sequence,
        ar_mask,
    ) = decoder_data
    b_node_features = jnp.stack([node_features] * 2)
    b_edge_features = jnp.stack([edge_features] * 2)
    b_neighbor_indices = jnp.stack([neighbor_indices] * 2)
    b_mask = jnp.stack([mask] * 2)
    b_attention_mask = jnp.stack([attention_mask] * 2)
    b_one_hot_sequence = jnp.stack([one_hot_sequence] * 2)
    b_ar_mask = jnp.stack([ar_mask] * 2)

    if decoding_approach == "unconditional":
        if attention_mask_type is None:
            decoder = jax.vmap(
                make_decoder(
                    model_parameters,
                    attention_mask_type=attention_mask_type,
                    decoding_approach=decoding_approach,
                ),
                in_axes=(0, 0, 0),
            )
            output = decoder(b_node_features, b_edge_features, b_mask)
        else:
            decoder = jax.vmap(
                make_decoder(
                    model_parameters,
                    attention_mask_type=attention_mask_type,
                    decoding_approach=decoding_approach,
                ),
                in_axes=(0, 0, 0, 0),
            )
            nodes_expanded = jnp.tile(
                jnp.expand_dims(b_node_features, -2),
                [1, 1, b_edge_features.shape[2], 1],
            )
            zeros_expanded = jnp.tile(
                jnp.expand_dims(jnp.zeros_like(b_node_features), -2),
                [1, 1, b_edge_features.shape[2], 1],
            )
            decoder_input_features = jnp.concatenate(
                [nodes_expanded, zeros_expanded, b_edge_features],
                -1,
            )
            output = decoder(
                b_node_features,
                decoder_input_features,
                b_mask,
                b_attention_mask,
            )
        chex.assert_shape(output, (2, 10, 128))

    elif decoding_approach == "conditional":
        decoder = jax.vmap(
            make_decoder(
                model_parameters,
                attention_mask_type=attention_mask_type,
                decoding_approach=decoding_approach,
            ),
            in_axes=(0, 0, 0, 0, 0, 0),
        )
        output = decoder(
            b_node_features,
            b_edge_features,
            b_neighbor_indices,
            b_mask,
            b_ar_mask,
            b_one_hot_sequence,
        )
        chex.assert_shape(output, (2, 10, 128))
