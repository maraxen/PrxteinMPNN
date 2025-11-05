import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.initialize import sampling_encode
from unittest.mock import patch


@pytest.fixture
def mock_model_parameters():
    return {}


def decoding_order_fn(key, n_nodes):
    return jnp.arange(n_nodes), jax.random.split(key)[0]


@patch("prxteinmpnn.sampling.initialize.project_features")
@patch("prxteinmpnn.sampling.initialize.extract_features")
def test_sampling_encode(
    mock_extract_features,
    mock_project_features,
    mock_model_parameters,
):
    mock_extract_features.return_value = (
        jnp.zeros((10, 48, 128)),
        jnp.zeros((10, 48), dtype=jnp.int32),
        jax.random.PRNGKey(0),
    )
    mock_project_features.return_value = jnp.zeros((10, 48, 128))

    def mock_encoder(edge_features, neighbor_indices, mask, attention_mask):
        return jnp.zeros((10, 128)), jnp.zeros((10, 48, 128))

    sample_model_pass_fn = sampling_encode(
        encoder=mock_encoder,
        decoding_order_fn=decoding_order_fn,
    )
    assert callable(sample_model_pass_fn)

    key = jax.random.PRNGKey(0)
    coords = jnp.zeros((10, 14, 3))
    mask = jnp.ones(10)
    residue_index = jnp.arange(10)
    chain_index = jnp.zeros(10, dtype=jnp.int32)

    output = sample_model_pass_fn(
        key,
        mock_model_parameters,
        coords,
        mask,
        residue_index,
        chain_index,
    )
    assert len(output) == 6
    node_features, edge_features, neighbor_indices, out_mask, ar_mask, out_key = output
    assert node_features.shape == (10, 128)
    assert edge_features.shape == (10, 48, 128)
    assert neighbor_indices.shape == (10, 48)
    assert out_mask.shape == (10,)
    assert ar_mask.shape == (10, 10)
    assert out_key.shape == (2,)
