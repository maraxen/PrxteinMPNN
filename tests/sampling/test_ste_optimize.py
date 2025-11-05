import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn
from unittest.mock import patch


def decoding_order_fn(key, n_nodes):
    return jnp.arange(n_nodes), jax.random.split(key)[0]


@patch("prxteinmpnn.sampling.ste_optimize.final_projection")
def test_make_optimize_sequence_fn(mock_final_projection):
    def mock_decoder(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        ar_mask,
        one_hot_sequence,
    ):
        return jnp.zeros((4, 10, 128))

    mock_final_projection.return_value = jnp.zeros((10, 21))

    model_parameters = {}
    optimize_sequence_fn = make_optimize_sequence_fn(
        mock_decoder, decoding_order_fn, model_parameters
    )
    assert callable(optimize_sequence_fn)

    key = jax.random.PRNGKey(0)
    node_features = jnp.zeros((10, 128))
    edge_features = jnp.zeros((10, 48, 128))
    neighbor_indices = jnp.zeros((10, 48), dtype=jnp.int32)
    mask = jnp.ones(10)
    iterations = 2
    learning_rate = 1e-4
    temperature = 1.0

    one_hot_sequence, logits = optimize_sequence_fn(
        key,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        iterations,
        learning_rate,
        temperature,
    )

    assert one_hot_sequence.shape == (10, 21)
    assert logits.shape == (10, 21)


@patch("prxteinmpnn.sampling.ste_optimize.final_projection")
def test_optimize_sequence_fn_differentiability(mock_final_projection):
    def mock_decoder(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        ar_mask,
        one_hot_sequence,
    ):
        return jnp.zeros((4, 10, 128))

    mock_final_projection.return_value = jnp.zeros((10, 21))

    model_parameters = {}
    optimize_sequence_fn = make_optimize_sequence_fn(
        mock_decoder, decoding_order_fn, model_parameters
    )

    key = jax.random.PRNGKey(0)
    node_features = jnp.zeros((10, 128))
    edge_features = jnp.zeros((10, 48, 128))
    neighbor_indices = jnp.zeros((10, 48), dtype=jnp.int32)
    mask = jnp.ones(10)
    iterations = 2
    learning_rate = 1e-4
    temperature = 1.0

    def loss_fn(logits):
        one_hot_sequence, _ = optimize_sequence_fn(
            key,
            node_features,
            edge_features,
            neighbor_indices,
            mask,
            iterations,
            learning_rate,
            temperature,
        )
        return jnp.sum(one_hot_sequence * logits)

    grads = jax.grad(loss_fn)(jnp.zeros((10, 21)))
    assert grads.shape == (10, 21)
