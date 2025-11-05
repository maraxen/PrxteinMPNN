
"""Tests for prxteinmpnn.sampling.sample."""

import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.sampling import (
    make_encoding_sampling_split_fn,
    make_sample_sequences,
    sample
)
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.data_structures import Protein


def test_make_sample_sequences_temperature(
    mock_model_parameters, model_inputs, rng_key
):
    """Test temperature sampling with make_sample_sequences."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    assert seq.dtype == jnp.int8
    assert seq.shape == (model_inputs["mask"].shape[0],)
    assert logits.shape == (
        model_inputs["mask"].shape[0],
        21,
    )
    assert order.shape == (model_inputs["mask"].shape[0],)


def test_make_encoding_sampling_split_fn(
    mock_model_parameters, model_inputs, rng_key
):
    """Test make_encoding_sampling_split_fn."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    encode_fn, sample_fn = make_encoding_sampling_split_fn(model)

    # Test encode_fn
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )
    assert isinstance(encoded_features, tuple)

    # Test sample_fn
    decoding_order, _ = random_decoding_order(
        rng_key, model_inputs["structure_coordinates"].shape[0]
    )
    seq = sample_fn(rng_key, encoded_features, decoding_order)

    assert seq.dtype == jnp.int8
    assert seq.shape == (model_inputs["mask"].shape[0],)


def test_make_sample_sequences_invalid_strategy(mock_model_parameters, rng_key):
    """Test make_sample_sequences with an invalid sampling strategy."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    with pytest.raises(ValueError):
        make_sample_sequences(model, sampling_strategy="invalid_strategy")


def test_make_sample_sequences_straight_through(
    mock_model_parameters, model_inputs, rng_key
):
    """Test straight_through sampling with make_sample_sequences."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    sample_fn = make_sample_sequences(model, sampling_strategy="straight_through")
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        iterations=10,
    )

    assert seq.dtype == jnp.int8
    assert seq.shape == (model_inputs["mask"].shape[0],)
    assert logits.shape == (
        model_inputs["mask"].shape[0],
        21,
    )
    assert order.shape == (model_inputs["mask"].shape[0],)


def test_sample_convenience_function(mock_model_parameters, model_inputs, rng_key):
    """Test the `sample` convenience function."""
    model = PrxteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_features=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        key=rng_key,
    )
    seq, logits, order = sample(
        rng_key,
        model,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    assert seq.dtype == jnp.int8
    assert seq.shape == (model_inputs["mask"].shape[0],)
    assert logits.shape == (
        model_inputs["mask"].shape[0],
        21,
    )
    assert order.shape == (model_inputs["mask"].shape[0],)
