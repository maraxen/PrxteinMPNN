import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.conditional_logits import (
    make_conditional_logits_fn,
    make_encoding_conditional_logits_split_fn,
)
from unittest.mock import patch


@pytest.fixture
def mock_model_parameters():
    return {}


def decoding_order_fn(key, n_nodes):
    return jnp.arange(n_nodes), jax.random.split(key)[0]


@patch("prxteinmpnn.sampling.conditional_logits.make_encoder")
@patch("prxteinmpnn.sampling.conditional_logits.make_decoder")
@patch("prxteinmpnn.sampling.conditional_logits.final_projection")
@patch("prxteinmpnn.sampling.initialize.project_features")
@patch("prxteinmpnn.sampling.initialize.extract_features")
def test_make_conditional_logits_fn(
    mock_extract_features,
    mock_project_features,
    mock_final_projection,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
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

    conditional_logits_fn = make_conditional_logits_fn(
        mock_model_parameters, decoding_order_fn=decoding_order_fn
    )
    assert callable(conditional_logits_fn)

    key = jax.random.PRNGKey(0)
    coords = jnp.zeros((10, 14, 3))
    seq = jnp.zeros(10, dtype=jnp.int32)
    mask = jnp.ones(10)
    residue_index = jnp.arange(10)
    chain_index = jnp.zeros(10, dtype=jnp.int32)

    logits, _, _ = conditional_logits_fn(
        key, coords, seq, mask, residue_index, chain_index
    )
    assert logits.shape == (10, 21)


@patch("prxteinmpnn.sampling.conditional_logits.make_encoder")
@patch("prxteinmpnn.sampling.conditional_logits.make_decoder")
@patch("prxteinmpnn.sampling.conditional_logits.final_projection")
@patch("prxteinmpnn.sampling.initialize.project_features")
@patch("prxteinmpnn.sampling.initialize.extract_features")
def test_conditional_logits_bias(
    mock_extract_features,
    mock_project_features,
    mock_final_projection,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
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

    conditional_logits_fn = make_conditional_logits_fn(
        mock_model_parameters, decoding_order_fn=decoding_order_fn
    )

    key = jax.random.PRNGKey(0)
    coords = jnp.zeros((10, 14, 3))
    seq = jnp.zeros(10, dtype=jnp.int32)
    mask = jnp.ones(10)
    residue_index = jnp.arange(10)
    chain_index = jnp.zeros(10, dtype=jnp.int32)
    bias = jax.nn.one_hot(jnp.arange(10), 21) * 10

    logits, _, _ = conditional_logits_fn(
        key, coords, seq, mask, residue_index, chain_index, bias=bias
    )

    assert jnp.allclose(logits, bias, atol=1e-6)


@patch("prxteinmpnn.sampling.conditional_logits.make_encoder")
@patch("prxteinmpnn.sampling.conditional_logits.make_decoder")
@patch("prxteinmpnn.sampling.conditional_logits.final_projection")
@patch("prxteinmpnn.sampling.initialize.project_features")
@patch("prxteinmpnn.sampling.initialize.extract_features")
def test_make_encoding_conditional_logits_split_fn(
    mock_extract_features,
    mock_project_features,
    mock_final_projection,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
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

    encode_fn, condition_logits_fn = make_encoding_conditional_logits_split_fn(
        mock_model_parameters, decoding_order_fn=decoding_order_fn
    )
    assert callable(encode_fn)
    assert callable(condition_logits_fn)

    key = jax.random.PRNGKey(0)
    coords = jnp.zeros((10, 14, 3))
    mask = jnp.ones(10)
    residue_index = jnp.arange(10)
    chain_index = jnp.zeros(10, dtype=jnp.int32)

    encoded_features = encode_fn(key, coords, mask, residue_index, chain_index)
    assert len(encoded_features) == 5

    seq = jnp.zeros(10, dtype=jnp.int32)
    logits, _, _ = condition_logits_fn(*encoded_features, seq)
    assert logits.shape == (10, 21)
