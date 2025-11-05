import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.sample import (
    make_sample_sequences,
    make_encoding_sampling_split_fn,
)
from prxteinmpnn.eqx_new import PrxteinMPNN
from unittest.mock import patch


class MockPrxteinMPNN(PrxteinMPNN):
    def __init__(self, key):
        super().__init__(
            node_features=128,
            edge_features=128,
            hidden_features=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=48,
            key=key,
        )

    def encoder(self, edge_features, neighbor_indices, mask):
        return jnp.zeros((10, 128)), jnp.zeros((10, 48, 128))

    def decoder(self, *args, **kwargs):
        return jnp.zeros((10, 128))


@pytest.fixture
def mock_equinox_model():
    return MockPrxteinMPNN(key=jax.random.PRNGKey(0))


def decoding_order_fn(key, n_nodes):
    return jnp.arange(n_nodes), jax.random.split(key)[0]


@patch("prxteinmpnn.sampling.sample.preload_sampling_step_decoder")
@patch("prxteinmpnn.sampling.sample.make_optimize_sequence_fn")
@patch("prxteinmpnn.sampling.sample.sampling_encode")
def test_make_sample_sequences(
    mock_sampling_encode,
    mock_make_optimize_sequence_fn,
    mock_preload_sampling_step_decoder,
    mock_equinox_model,
):
    mock_sampling_encode.return_value = lambda *args, **kwargs: (
        jnp.zeros((10, 128)),
        jnp.zeros((10, 48, 128)),
        jnp.zeros((10, 48), dtype=jnp.int32),
        jnp.arange(10),
        jnp.ones((10, 10)),
        jax.random.PRNGKey(0),
    )
    mock_make_optimize_sequence_fn.return_value = lambda *args, **kwargs: (
        jax.nn.one_hot(jnp.zeros(10, dtype=jnp.int32), 21),
        jnp.zeros((10, 21)),
    )
    mock_preload_sampling_step_decoder.return_value = lambda *args, **kwargs: (
        None,
        jnp.zeros(10, dtype=jnp.int32),
        jnp.zeros((10, 21)),
    )

    # Test with temperature sampling
    sample_sequences_fn = make_sample_sequences(
        mock_equinox_model,
        decoding_order_fn,
        sampling_strategy="temperature",
    )
    assert callable(sample_sequences_fn)

    key = jax.random.PRNGKey(0)
    coords = jnp.zeros((10, 14, 3))
    mask = jnp.ones(10)
    residue_index = jnp.arange(10)
    chain_index = jnp.zeros(10, dtype=jnp.int32)

    seq, logits, decoding_order = sample_sequences_fn(
        key, coords, mask, residue_index, chain_index
    )
    assert seq.shape == (10,)
    assert logits.shape == (10, 21)
    assert decoding_order.shape == (10,)

    # Test with straight-through sampling
    sample_sequences_fn = make_sample_sequences(
        mock_equinox_model,
        decoding_order_fn,
        sampling_strategy="straight_through",
    )
    assert callable(sample_sequences_fn)

    seq, logits, decoding_order = sample_sequences_fn(
        key, coords, mask, residue_index, chain_index
    )
    assert seq.shape == (10,)
    assert logits.shape == (10, 21)
    assert decoding_order.shape == (10,)


@patch("prxteinmpnn.sampling.sample.preload_sampling_step_decoder")
@patch("prxteinmpnn.sampling.sample.make_optimize_sequence_fn")
@patch("prxteinmpnn.sampling.sample.sampling_encode")
def test_sample_sequences_determinism(
    mock_sampling_encode,
    mock_make_optimize_sequence_fn,
    mock_preload_sampling_step_decoder,
    mock_equinox_model,
):
    mock_sampling_encode.return_value = lambda *args, **kwargs: (
        jnp.zeros((10, 128)),
        jnp.zeros((10, 48, 128)),
        jnp.zeros((10, 48), dtype=jnp.int32),
        jnp.arange(10),
        jnp.ones((10, 10)),
        jax.random.PRNGKey(0),
    )
    mock_make_optimize_sequence_fn.return_value = lambda *args, **kwargs: (
        jax.nn.one_hot(jnp.zeros(10, dtype=jnp.int32), 21),
        jnp.zeros((10, 21)),
    )
    mock_preload_sampling_step_decoder.return_value = lambda *args, **kwargs: (
        None,
        jnp.zeros(10, dtype=jnp.int32),
        jnp.zeros((10, 21)),
    )

    sample_sequences_fn = make_sample_sequences(
        mock_equinox_model,
        decoding_order_fn,
        sampling_strategy="temperature",
    )

    key = jax.random.PRNGKey(0)
    coords = jnp.zeros((10, 14, 3))
    mask = jnp.ones(10)
    residue_index = jnp.arange(10)
    chain_index = jnp.zeros(10, dtype=jnp.int32)

    seq1, _, _ = sample_sequences_fn(
        key, coords, mask, residue_index, chain_index, temperature=0.0
    )
    seq2, _, _ = sample_sequences_fn(
        key, coords, mask, residue_index, chain_index, temperature=0.0
    )

    assert jnp.array_equal(seq1, seq2)


@patch("prxteinmpnn.sampling.sample.preload_sampling_step_decoder")
@patch("prxteinmpnn.sampling.sample.make_optimize_sequence_fn")
@patch("prxteinmpnn.sampling.sample.sampling_encode")
def test_make_encoding_sampling_split_fn(
    mock_sampling_encode,
    mock_make_optimize_sequence_fn,
    mock_preload_sampling_step_decoder,
    mock_equinox_model,
):
    mock_sampling_encode.return_value = lambda *args, **kwargs: (
        jnp.zeros((10, 128)),
        jnp.zeros((10, 48, 128)),
        jnp.zeros((10, 48), dtype=jnp.int32),
        jnp.arange(10),
        jnp.ones((10, 10)),
        jax.random.PRNGKey(0),
    )
    mock_make_optimize_sequence_fn.return_value = lambda *args, **kwargs: (
        jax.nn.one_hot(jnp.zeros(10, dtype=jnp.int32), 21),
        jnp.zeros((10, 21)),
    )
    mock_preload_sampling_step_decoder.return_value = lambda *args, **kwargs: (
        None,
        jnp.zeros(10, dtype=jnp.int32),
        jnp.zeros((10, 21)),
    )

    encode_fn, sample_from_features_fn = make_encoding_sampling_split_fn(
        mock_equinox_model,
        decoding_order_fn,
    )
    assert callable(encode_fn)
    assert callable(sample_from_features_fn)

    key = jax.random.PRNGKey(0)
    coords = jnp.zeros((10, 14, 3))
    mask = jnp.ones(10)
    residue_index = jnp.arange(10)
    chain_index = jnp.zeros(10, dtype=jnp.int32)

    encoded_features, decoding_order = encode_fn(
        key, coords, mask, residue_index, chain_index
    )
    assert "node_features" in encoded_features
    assert "edge_features" in encoded_features
    assert "neighbor_indices" in encoded_features
    assert "mask" in encoded_features
    assert decoding_order.shape == (10,)

    seq, logits = sample_from_features_fn(key, encoded_features, decoding_order)
    assert seq.shape == (10,)
    assert logits.shape == (10, 21)
