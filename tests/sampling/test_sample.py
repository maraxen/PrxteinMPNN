
"""Tests for prxteinmpnn.sampling.sample."""

import chex
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.run.averaging import make_encoding_sampling_split_fn
from prxteinmpnn.sampling import (
    make_sample_sequences,
    sample,
)
from prxteinmpnn.utils.decoding_order import random_decoding_order


def test_make_sample_sequences_temperature_jit(
    mock_model_parameters, model_inputs, rng_key,
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
    sample_fn = jax.jit(
        make_sample_sequences(model, sampling_strategy="temperature"),
    )
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_make_sample_sequences_temperature_no_jit(
    mock_model_parameters, model_inputs, rng_key,
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

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_make_encoding_sampling_split_fn_jit(
    mock_model_parameters, model_inputs, rng_key,
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
    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)
    encode_fn = jax.jit(encode_fn)
    sample_fn = jax.jit(sample_fn)

    # Test encode_fn
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )
    assert isinstance(encoded_features, tuple)
    chex.assert_tree_all_finite(encoded_features)

    # Test sample_fn
    decoding_order, _ = random_decoding_order(
        rng_key, model_inputs["structure_coordinates"].shape[0],
    )
    seq = sample_fn(rng_key, encoded_features, decoding_order)

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite(seq)


def test_make_encoding_sampling_split_fn_no_jit(
    mock_model_parameters, model_inputs, rng_key,
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
    encode_fn, sample_fn, _ = make_encoding_sampling_split_fn(model)

    # Test encode_fn
    encoded_features = encode_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )
    assert isinstance(encoded_features, tuple)
    chex.assert_tree_all_finite(encoded_features)

    # Test sample_fn
    decoding_order, _ = random_decoding_order(
        rng_key, model_inputs["structure_coordinates"].shape[0],
    )
    seq = sample_fn(rng_key, encoded_features, decoding_order)

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite(seq)


def test_make_sample_sequences_invalid_strategy(
    mock_model_parameters, rng_key,
):
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
    with pytest.raises(ValueError, match="Unknown sampling strategy"):
        make_sample_sequences(model, sampling_strategy="invalid_strategy")


def test_make_sample_sequences_straight_through_jit(
    mock_model_parameters, model_inputs, rng_key,
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
    sample_fn = jax.jit(
        make_sample_sequences(model, sampling_strategy="straight_through"),
        static_argnames=["iterations"],
    )
    seq, logits, order = sample_fn(
        rng_key,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
        iterations=10,
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_make_sample_sequences_straight_through_no_jit(
    mock_model_parameters, model_inputs, rng_key,
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

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_sample_convenience_function_jit(
    mock_model_parameters, model_inputs, rng_key,
):
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
    sample_fn = jax.jit(sample, static_argnames=["model"])
    seq, logits, order = sample_fn(
        rng_key,
        model,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))


def test_sample_convenience_function_no_jit(
    mock_model_parameters, model_inputs, rng_key,
):
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
    sample_fn = sample
    seq, logits, order = sample_fn(
        rng_key,
        model,
        model_inputs["structure_coordinates"],
        model_inputs["mask"],
        model_inputs["residue_index"],
        model_inputs["chain_index"],
    )

    chex.assert_type(seq, jnp.int8)
    chex.assert_shape(seq, (model_inputs["mask"].shape[0],))
    chex.assert_shape(logits, (model_inputs["mask"].shape[0], 21))
    chex.assert_shape(order, (model_inputs["mask"].shape[0],))
    chex.assert_tree_all_finite((seq, logits, order))
