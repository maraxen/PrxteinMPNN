# type: ignore[call-arg]
"""Integration tests for the sampling factory."""

import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.sample import make_sample_sequences


def test_make_sample_sequences(mock_model_parameters, model_inputs, rng_key):
    """Test the full sequence sampling pipeline.

    Raises:
        AssertionError: If the output does not match expected shapes or properties.
    """
    L, K = model_inputs["sequence"].shape[0], 48
    key = rng_key

    decoding_order_fn = lambda k, l: (
        jax.lax.iota(jnp.int32, l),
        jax.random.split(k)[1],
    )

    sample_sequences_fn = make_sample_sequences(
        model_parameters=mock_model_parameters,
        decoding_order_fn=decoding_order_fn,
        sampling_strategy="temperature",
        num_encoder_layers=3,
        num_decoder_layers=3,
    )

    # Prepare inputs
    key, sample_key = jax.random.split(key)

    # Run sampling
    sampled_sequence, logits, decoding_order = sample_sequences_fn(
        prng_key=sample_key,
        structure_coordinates=model_inputs["structure_coordinates"],
        mask=jnp.ones((L,), dtype=jnp.bool_),
        residue_index=model_inputs["residue_index"],
        chain_index=model_inputs["chain_index"],
        k_neighbors=K,
    )

    # Check output shapes
    chex.assert_shape(sampled_sequence, (L,))
    chex.assert_shape(logits, (L, 21))
    chex.assert_shape(decoding_order, (L,))

    assert (
        jnp.unique(sampled_sequence).shape[0] <= 21
    ), "Sampled sequence contains invalid amino acid indices."