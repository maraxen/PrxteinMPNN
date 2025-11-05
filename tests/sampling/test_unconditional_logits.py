import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn
from unittest.mock import patch


def decoding_order_fn(key, n_nodes):
    return jnp.arange(n_nodes), jax.random.split(key)[0]


@patch("prxteinmpnn.sampling.unconditional_logits.final_projection")
@patch("prxteinmpnn.sampling.unconditional_logits.make_decoder")
@patch("prxteinmpnn.sampling.unconditional_logits.make_encoder")
@patch("prxteinmpnn.sampling.initialize.project_features")
@patch("prxteinmpnn.sampling.initialize.extract_features")
def test_make_unconditional_logits_fn(
    mock_extract_features,
    mock_project_features,
    mock_make_encoder,
    mock_make_decoder,
    mock_final_projection,
):
    mock_extract_features.return_value = (
        jnp.zeros((10, 48, 128)),
        jnp.zeros((10, 48), dtype=jnp.int32),
        jax.random.PRNGKey(0),
    )
    mock_project_features.return_value = jnp.zeros((10, 48, 128))
    mock_make_encoder.return_value = lambda *args, **kwargs: (
        jnp.zeros((10, 128)),
        jnp.zeros((10, 48, 128)),
    )
    mock_make_decoder.return_value = lambda *args, **kwargs: jnp.zeros((10, 128))
    mock_final_projection.return_value = jnp.zeros((10, 21))

    model_parameters = {}
    unconditional_logits_fn = make_unconditional_logits_fn(
        model_parameters, decoding_order_fn=decoding_order_fn
    )
    assert callable(unconditional_logits_fn)

    key = jax.random.PRNGKey(0)
    coords = jnp.zeros((10, 14, 3))
    mask = jnp.ones(10)
    residue_index = jnp.arange(10)
    chain_index = jnp.zeros(10, dtype=jnp.int32)

    logits, node_features, edge_features = unconditional_logits_fn(
        key, coords, mask, residue_index, chain_index
    )

    assert logits.shape == (10, 21)
    assert node_features.shape == (10, 128)
    assert edge_features.shape == (10, 48, 128)
