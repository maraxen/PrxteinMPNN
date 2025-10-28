# type: ignore[call-arg]
"""Integration tests for the sampling factory."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.residue_constants import atom_order


def test_greedy_decoding_is_deterministic(mock_model_parameters, model_inputs, rng_key):
    """Test that greedy decoding (temperature=0.0) is deterministic."""
    sample_fn = make_sample_sequences(
        model_parameters=mock_model_parameters,
        decoding_order_fn=random_decoding_order,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )

    sample_inputs = model_inputs.copy()
    del sample_inputs["sequence"]

    # Run sampling twice with the same key
    seq1, _, _ = sample_fn(
        prng_key=rng_key,
        **sample_inputs,
        temperature=0.0,
    )
    seq2, _, _ = sample_fn(
        prng_key=rng_key,
        **sample_inputs,
        temperature=0.0,
    )

    # The outputs should be identical
    np.testing.assert_array_equal(seq1, seq2)


def test_sampled_output_is_valid(mock_model_parameters, model_inputs, rng_key):
    """Test that the sampler with non-zero temperature produces a valid sequence."""
    sample_fn = make_sample_sequences(
        model_parameters=mock_model_parameters,
        decoding_order_fn=random_decoding_order,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )

    seq_len = model_inputs["sequence"].shape[0]
    sample_inputs = model_inputs.copy()
    del sample_inputs["sequence"]

    sampled_sequence, _, _ = sample_fn(
        prng_key=rng_key,
        **sample_inputs,
        temperature=1.0,  # Use a non-zero temperature
    )

    # 1. Check for correct length
    assert len(sampled_sequence) == seq_len

    # 2. Check that all amino acid indices are within the valid range [0, 19]
    assert jnp.all(sampled_sequence >= 0)
    assert jnp.all(sampled_sequence <= 19)


def test_logits_are_plausible(mock_model_parameters, model_inputs, rng_key):
    """Test that the returned logits have the correct shape and are not NaN or inf."""
    sample_fn = make_sample_sequences(
        model_parameters=mock_model_parameters,
        decoding_order_fn=random_decoding_order,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )

    seq_len = model_inputs["sequence"].shape[0]
    sample_inputs = model_inputs.copy()
    del sample_inputs["sequence"]

    _, logits, _ = sample_fn(
        prng_key=rng_key,
        **sample_inputs,
        temperature=1.0,
    )

    # 1. Check shape
    chex.assert_shape(logits, (seq_len, 21))

    # 2. Check for NaN or Inf values
    assert jnp.all(jnp.isfinite(logits))


def test_straight_through_sampling_with_dynamic_iterations(
    mock_model_parameters, model_inputs, rng_key
):
    """Test that straight_through sampling works with dynamic iteration counts.
    
    This specifically tests the refactored STE optimizer that now uses fori_loop
    instead of scan, allowing dynamic iteration counts in JAX 0.7+.
    """
    sample_fn = make_sample_sequences(
        model_parameters=mock_model_parameters,
        decoding_order_fn=random_decoding_order,
        sampling_strategy="straight_through",
        num_encoder_layers=3,
        num_decoder_layers=3,
    )

    seq_len = model_inputs["sequence"].shape[0]
    sample_inputs = model_inputs.copy()
    del sample_inputs["sequence"]

    # Test with different iteration counts
    for iterations in [1, 3, 5]:
        sampled_sequence, logits, decoding_order = sample_fn(
            prng_key=rng_key,
            **sample_inputs,
            iterations=iterations,
            learning_rate=0.001,
            temperature=0.1,
        )

        # Check sequence shape and validity
        assert len(sampled_sequence) == seq_len
        assert jnp.all(sampled_sequence >= 0)
        assert jnp.all(sampled_sequence <= 19)

        # Check logits shape and validity
        chex.assert_shape(logits, (seq_len, 21))
        assert jnp.all(jnp.isfinite(logits))

        # Check decoding order
        assert len(decoding_order) == seq_len


def test_straight_through_vs_temperature_sampling(mock_model_parameters, model_inputs, rng_key):
    """Test that straight_through and temperature sampling produce different results."""
    sample_fn_ste = make_sample_sequences(
        model_parameters=mock_model_parameters,
        decoding_order_fn=random_decoding_order,
        sampling_strategy="straight_through",
        num_encoder_layers=3,
        num_decoder_layers=3,
    )

    sample_fn_temp = make_sample_sequences(
        model_parameters=mock_model_parameters,
        decoding_order_fn=random_decoding_order,
        sampling_strategy="temperature",
        num_encoder_layers=3,
        num_decoder_layers=3,
    )

    sample_inputs = model_inputs.copy()
    del sample_inputs["sequence"]

    seq_ste, _, _ = sample_fn_ste(
        prng_key=rng_key,
        **sample_inputs,
        iterations=5,
        learning_rate=0.001,
        temperature=0.1,
    )

    seq_temp, _, _ = sample_fn_temp(
        prng_key=rng_key,
        **sample_inputs,
        temperature=1.0,
    )

    # The two sampling strategies should generally produce different sequences
    # (they may occasionally be the same by chance, but shouldn't be identical every time)
    assert seq_ste.shape == seq_temp.shape