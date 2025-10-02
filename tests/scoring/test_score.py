"""Tests for the scoring module."""

from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from prxteinmpnn.scoring.score import make_score_sequence


@pytest.fixture
def mock_decoding_order_fn():
    """Create mock decoding order function."""

    def mock_fn(prng_key: PRNGKeyArray, seq_len: int):
        return jnp.arange(seq_len), prng_key

    return mock_fn


@patch("prxteinmpnn.scoring.score.make_encoder")
@patch("prxteinmpnn.scoring.score.make_decoder")
@patch("prxteinmpnn.scoring.score.extract_features")
@patch("prxteinmpnn.scoring.score.project_features")
@patch("prxteinmpnn.scoring.score.final_projection")
@patch("prxteinmpnn.scoring.score.generate_ar_mask")
def test_make_score_sequence_without_model_inputs(
    mock_generate_ar_mask,
    mock_final_projection,
    mock_project_features,
    mock_extract_features,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
    mock_decoding_order_fn,
    model_inputs,
    rng_key,
):
    """Test make_score_sequence without model inputs returns base scoring function."""
    # Setup mocks
    mock_encoder = Mock()
    mock_decoder = Mock()
    mock_make_encoder.return_value = mock_encoder
    mock_make_decoder.return_value = mock_decoder

    seq_len = model_inputs["sequence"].shape[0]
    mock_generate_ar_mask.return_value = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
    # Fix: neighbor_indices should be integer type
    mock_extract_features.return_value = (
        jnp.ones((seq_len, 10)),  # edge_features
        jnp.ones((seq_len, 48), dtype=jnp.int32),  # neighbor_indices
        rng_key,  # prng_key
    )
    mock_project_features.return_value = jnp.ones((seq_len, 10))
    mock_encoder.return_value = (jnp.ones((seq_len, 128)), jnp.ones((seq_len, 10)))
    mock_decoder.return_value = jnp.ones((seq_len, 128))
    mock_final_projection.return_value = jnp.ones((seq_len, 20))

    # Create scoring function
    scoring_fn = make_score_sequence(
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
        num_encoder_layers=3,
        num_decoder_layers=3,
    )

    # Test scoring function
    score, logits, decoding_order = scoring_fn(
        prng_key=rng_key,
        sequence=model_inputs["sequence"],
        structure_coordinates=model_inputs["structure_coordinates"],
        mask=model_inputs["mask"],
        residue_index=model_inputs["residue_index"],
        chain_index=model_inputs["chain_index"],
        k_neighbors=48,
        backbone_noise=0.0,
        ar_mask=jnp.ones((seq_len, seq_len), dtype=jnp.bool_),
    )

    # Verify result shape and type
    chex.assert_shape(score, ())  # Should be a scalar score
    chex.assert_type(score, jnp.floating)
    chex.assert_shape(logits, (seq_len, 20))
    chex.assert_type(logits, jnp.floating)
    chex.assert_shape(decoding_order, (seq_len,))
    chex.assert_type(decoding_order, jnp.int32)


@patch("prxteinmpnn.scoring.score.make_encoder")
@patch("prxteinmpnn.scoring.score.make_decoder")
def test_encoder_decoder_creation(
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
    mock_decoding_order_fn,
):
    """Test that encoder and decoder are created with correct parameters."""
    from prxteinmpnn.model.decoder import DecodingApproach
    from prxteinmpnn.model.masked_attention import MaskedAttentionType

    make_score_sequence(
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
        num_encoder_layers=5,
        num_decoder_layers=7,
    )

    # Verify encoder creation
    mock_make_encoder.assert_called_once_with(
        model_parameters=mock_model_parameters,
        attention_mask_type="cross",
        num_encoder_layers=5,
    )

    # Verify decoder creation
    mock_make_decoder.assert_called_once_with(
        model_parameters=mock_model_parameters,
        attention_mask_type=None,
        decoding_approach="conditional",
        num_decoder_layers=7,
    )


@patch("prxteinmpnn.scoring.score.make_encoder")
@patch("prxteinmpnn.scoring.score.make_decoder")
@patch("prxteinmpnn.scoring.score.extract_features")
@patch("prxteinmpnn.scoring.score.project_features")
@patch("prxteinmpnn.scoring.score.final_projection")
@patch("prxteinmpnn.scoring.score.generate_ar_mask")
def test_jit_compilation(
    mock_generate_ar_mask,
    mock_final_projection,
    mock_project_features,
    mock_extract_features,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
    mock_decoding_order_fn,
    model_inputs,
    rng_key,
):
    """Test that the scoring function can be JIT compiled."""
    # Setup mocks
    mock_encoder = Mock()
    mock_decoder = Mock()
    mock_make_encoder.return_value = mock_encoder
    mock_make_decoder.return_value = mock_decoder

    seq_len = model_inputs["sequence"].shape[0]
    mock_generate_ar_mask.return_value = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
    # Fix: neighbor_indices should be integer type
    mock_extract_features.return_value = (
        jnp.ones((seq_len, 10)),
        jnp.ones((seq_len, 48), dtype=jnp.int32),
        rng_key,
    )
    mock_project_features.return_value = jnp.ones((seq_len, 10))
    mock_encoder.return_value = (jnp.ones((seq_len, 128)), jnp.ones((seq_len, 10)))
    mock_decoder.return_value = jnp.ones((seq_len, 128))
    mock_final_projection.return_value = jnp.ones((seq_len, 20))

    scoring_fn = make_score_sequence(
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
    )

    # Compile the function
    compiled_fn = jax.jit(scoring_fn, static_argnames=("k_neighbors",))

    # Test that compiled function works
    score, logits, decoding_order = compiled_fn(
        prng_key=rng_key,
        sequence=model_inputs["sequence"],
        structure_coordinates=model_inputs["structure_coordinates"],
        mask=model_inputs["mask"],
        residue_index=model_inputs["residue_index"],
        chain_index=model_inputs["chain_index"],
        k_neighbors=48,
        backbone_noise=0.0,
    )

    chex.assert_shape(score, ())
    chex.assert_type(score, jnp.floating)
    chex.assert_shape(logits, (seq_len, 20))
    chex.assert_type(logits, jnp.floating)
    chex.assert_shape(decoding_order, (seq_len,))
    chex.assert_type(decoding_order, jnp.int32)


@patch("prxteinmpnn.scoring.score.make_encoder")
@patch("prxteinmpnn.scoring.score.make_decoder")
@patch("prxteinmpnn.scoring.score.extract_features")
@patch("prxteinmpnn.scoring.score.project_features")
@patch("prxteinmpnn.scoring.score.final_projection")
@patch("prxteinmpnn.scoring.score.generate_ar_mask")
def test_different_sequence_lengths(
    mock_generate_ar_mask,
    mock_final_projection,
    mock_project_features,
    mock_extract_features,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
    mock_decoding_order_fn,
    rng_key,
):
    """Test scoring function with different sequence lengths."""
    # Setup mocks
    mock_encoder = Mock()
    mock_decoder = Mock()
    mock_make_encoder.return_value = mock_encoder
    mock_make_decoder.return_value = mock_decoder

    scoring_fn = make_score_sequence(
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
    )

    for seq_len in [5, 20, 76]:
        # Setup mocks for this sequence length
        mock_generate_ar_mask.return_value = jnp.ones(
            (seq_len, seq_len), dtype=jnp.bool_
        )
        mock_extract_features.return_value = (
            jnp.ones((seq_len, 10)),  # edge_features
            jnp.ones((seq_len, 48), dtype=jnp.int32),  # neighbor_indices
            rng_key,  # prng_key
        )
        mock_project_features.return_value = jnp.ones((seq_len, 10))
        mock_encoder.return_value = (
            jnp.ones((seq_len, 128)),
            jnp.ones((seq_len, 10)),
        )
        mock_decoder.return_value = jnp.ones((seq_len, 128))
        mock_final_projection.return_value = jnp.ones((seq_len, 20))

        # Create inputs for this sequence length
        sequence = jnp.ones((seq_len,), dtype=jnp.int32)
        structure_coordinates = jnp.ones((seq_len, 37, 3))
        mask = jnp.ones((seq_len, 37), dtype=jnp.bool_)
        residue_index = jnp.arange(seq_len)
        chain_index = jnp.zeros((seq_len,), dtype=jnp.int32)
        k_neighbors = 48
        backbone_noise = jnp.array(0.0)
        ar_mask = None

        score, logits, decoding_order = scoring_fn(
            rng_key,
            sequence,
            structure_coordinates,
            mask,
            residue_index,
            chain_index,
            k_neighbors,
            backbone_noise,
            ar_mask,
        )
        chex.assert_shape(score, ())
        chex.assert_type(score, jnp.floating)
        chex.assert_shape(logits, (seq_len, 20))
        chex.assert_type(logits, jnp.floating)
        chex.assert_shape(decoding_order, (seq_len,))
        chex.assert_type(decoding_order, jnp.int32)


def test_scoring_function_types(mock_model_parameters, mock_decoding_order_fn):
    """Test that the correct function types are returned."""
    # Test without model inputs
    scoring_fn_base = make_score_sequence(
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
    )

    # Should be callable with all parameters
    assert callable(scoring_fn_base)


@patch("prxteinmpnn.scoring.score.make_encoder")
@patch("prxteinmpnn.scoring.score.make_decoder")
@patch("prxteinmpnn.scoring.score.extract_features")
@patch("prxteinmpnn.scoring.score.project_features")
@patch("prxteinmpnn.scoring.score.final_projection")
@patch("prxteinmpnn.scoring.score.generate_ar_mask")
def test_scoring_output_calculation(
    mock_generate_ar_mask,
    mock_final_projection,
    mock_project_features,
    mock_extract_features,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
    mock_decoding_order_fn,
    model_inputs,
    rng_key,
):
    """Test the scoring output calculation logic."""
    # Setup mocks
    mock_encoder = Mock()
    mock_decoder = Mock()
    mock_make_encoder.return_value = mock_encoder
    mock_make_decoder.return_value = mock_decoder

    seq_len = model_inputs["sequence"].shape[0]
    vocab_size = 20

    # Mock final projection to return specific logits for testing
    test_logits = jnp.ones((seq_len, vocab_size)) * 2.0
    mock_final_projection.return_value = test_logits

    # Setup other mocks
    mock_generate_ar_mask.return_value = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
    # Fix: neighbor_indices should be integer type
    mock_extract_features.return_value = (
        jnp.ones((seq_len, 10)),
        jnp.ones((seq_len, 48), dtype=jnp.int32),
        rng_key,
    )
    mock_project_features.return_value = jnp.ones((seq_len, 10))
    mock_encoder.return_value = (jnp.ones((seq_len, 128)), jnp.ones((seq_len, 10)))
    mock_decoder.return_value = jnp.ones((seq_len, 128))

    scoring_fn = make_score_sequence(
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
    )

    score, logits, decoding_order = scoring_fn(
        prng_key=rng_key,
        sequence=model_inputs["sequence"],
        structure_coordinates=model_inputs["structure_coordinates"],
        mask=model_inputs["mask"],
        residue_index=model_inputs["residue_index"],
        chain_index=model_inputs["chain_index"],
        k_neighbors=48,
        backbone_noise=0.0,
        ar_mask=jnp.ones((seq_len, seq_len), dtype=jnp.bool_),
    )

    assert jnp.all(jnp.isfinite(score))
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.all(jnp.isfinite(decoding_order))