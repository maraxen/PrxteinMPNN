"""Tests for the scoring module."""

from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from prxteinmpnn.scoring.score import make_score_sequence


class TestMakeScoreSequence:
  """Test the make_score_sequence function."""

  @pytest.fixture
  def mock_model_parameters(self):
    """Create mock model parameters for testing."""
    # Create a more realistic mock that includes expected parameter structure
    params = {}
    for layer_idx in range(3):  # Default num_encoder_layers
      prefix = "protein_mpnn/~/enc_layer"
      if layer_idx > 0:
        prefix += f"_{layer_idx}"
      layer_suffix = f"enc{layer_idx}"
      
      params.update({
        f"{prefix}/~/{layer_suffix}_W1": jnp.ones((128, 128)),
        f"{prefix}/~/{layer_suffix}_W2": jnp.ones((128, 128)),
        f"{prefix}/~/{layer_suffix}_W3": jnp.ones((128, 128)),
        f"{prefix}/~/{layer_suffix}_norm1": {"scale": jnp.ones((128,)), "offset": jnp.zeros((128,))},
        f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_in": jnp.ones((128, 512)),
        f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_out": jnp.ones((512, 128)),
        f"{prefix}/~/{layer_suffix}_norm2": {"scale": jnp.ones((128,)), "offset": jnp.zeros((128,))},
        f"{prefix}/~/{layer_suffix}_W11": jnp.ones((128, 128)),
        f"{prefix}/~/{layer_suffix}_W12": jnp.ones((128, 128)),
        f"{prefix}/~/{layer_suffix}_W13": jnp.ones((128, 128)),
        f"{prefix}/~/{layer_suffix}_norm3": {"scale": jnp.ones((128,)), "offset": jnp.zeros((128,))},
      })
    
    # Add decoder parameters
    for layer_idx in range(3):  # Default num_decoder_layers
      prefix = "protein_mpnn/~/dec_layer"
      if layer_idx > 0:
        prefix += f"_{layer_idx}"
      layer_suffix = f"dec{layer_idx}"
      
      params.update({
        f"{prefix}/~/{layer_suffix}_W1": jnp.ones((128, 128)),
        f"{prefix}/~/{layer_suffix}_W2": jnp.ones((128, 128)),
        f"{prefix}/~/{layer_suffix}_W3": jnp.ones((128, 128)),
        f"{prefix}/~/{layer_suffix}_norm1": {"scale": jnp.ones((128,)), "offset": jnp.zeros((128,))},
        f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_in": jnp.ones((128, 512)),
        f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_out": jnp.ones((512, 128)),
        f"{prefix}/~/{layer_suffix}_norm2": {"scale": jnp.ones((128,)), "offset": jnp.zeros((128,))},
      })
    
    return params

  @pytest.fixture
  def mock_decoding_order_fn(self):
    """Create mock decoding order function."""
    def mock_fn(prng_key: PRNGKeyArray, seq_len: int):
      return jnp.arange(seq_len), prng_key
    return mock_fn

  @pytest.fixture
  def sample_inputs(self):
    """Create sample inputs for testing."""
    seq_len = 10
    num_atoms = 37  # All atoms
    return {
      "prng_key": jax.random.PRNGKey(42),
      "sequence": jnp.ones((seq_len,), dtype=jnp.int32),
      "structure_coordinates": jnp.ones((seq_len, num_atoms, 3)),
      "mask": jnp.ones((seq_len, num_atoms), dtype=jnp.bool_),
      "residue_index": jnp.arange(seq_len),
      "chain_index": jnp.zeros((seq_len,), dtype=jnp.int32),
      "k_neighbors": 48,
      "backbone_noise": 0.0,
      "ar_mask": jnp.ones((seq_len, seq_len), dtype=jnp.bool_),
    }



  @patch("prxteinmpnn.scoring.score.make_encoder")
  @patch("prxteinmpnn.scoring.score.make_decoder")
  @patch("prxteinmpnn.scoring.score.extract_features")
  @patch("prxteinmpnn.scoring.score.project_features")
  @patch("prxteinmpnn.scoring.score.final_projection")
  @patch("prxteinmpnn.scoring.score.generate_ar_mask")
  def test_make_score_sequence_without_model_inputs(
    self,
    mock_generate_ar_mask,
    mock_final_projection,
    mock_project_features,
    mock_extract_features,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
    mock_decoding_order_fn,
    sample_inputs,
  ):
    """Test make_score_sequence without model inputs returns base scoring function."""
    # Setup mocks
    mock_encoder = Mock()
    mock_decoder = Mock()
    mock_make_encoder.return_value = mock_encoder
    mock_make_decoder.return_value = mock_decoder
    
    seq_len = sample_inputs["sequence"].shape[0]
    mock_generate_ar_mask.return_value = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
    # Fix: neighbor_indices should be integer type
    mock_extract_features.return_value = (
      jnp.ones((seq_len, 10)),  # edge_features
      jnp.ones((seq_len, 48), dtype=jnp.int32),  # neighbor_indices
      jax.random.PRNGKey(42),  # prng_key
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
      sample_inputs["prng_key"],
      sample_inputs["sequence"],
      sample_inputs["structure_coordinates"],
      sample_inputs["mask"],
      sample_inputs["residue_index"],
      sample_inputs["chain_index"],
      sample_inputs["k_neighbors"],
      sample_inputs["backbone_noise"],
      sample_inputs["ar_mask"],
    )

    # Verify result shape and type
    chex.assert_shape(score, ()) # Should be a scalar score
    chex.assert_type(score, jnp.floating)
    chex.assert_shape(logits, (seq_len, 20))
    chex.assert_type(logits, jnp.floating)
    chex.assert_shape(decoding_order, (seq_len,))
    chex.assert_type(decoding_order, jnp.int32)


  @patch("prxteinmpnn.scoring.score.make_encoder")
  @patch("prxteinmpnn.scoring.score.make_decoder")
  def test_encoder_decoder_creation(
    self,
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
    self,
    mock_generate_ar_mask,
    mock_final_projection,
    mock_project_features,
    mock_extract_features,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
    mock_decoding_order_fn,
    sample_inputs,
  ):
    """Test that the scoring function can be JIT compiled."""
    # Setup mocks
    mock_encoder = Mock()
    mock_decoder = Mock()
    mock_make_encoder.return_value = mock_encoder
    mock_make_decoder.return_value = mock_decoder
    
    seq_len = sample_inputs["sequence"].shape[0]
    mock_generate_ar_mask.return_value = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
    # Fix: neighbor_indices should be integer type
    mock_extract_features.return_value = (
      jnp.ones((seq_len, 10)), 
      jnp.ones((seq_len, 48), dtype=jnp.int32),
      jax.random.PRNGKey(42),  # prng_key
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
      sample_inputs["prng_key"],
      sample_inputs["sequence"],
      sample_inputs["structure_coordinates"],
      sample_inputs["mask"],
      sample_inputs["residue_index"],
      sample_inputs["chain_index"],
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
    self,
    mock_generate_ar_mask,
    mock_final_projection,
    mock_project_features,
    mock_extract_features,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
    mock_decoding_order_fn,
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

    for seq_len in [5, 20, 100]:
      # Setup mocks for this sequence length
      mock_generate_ar_mask.return_value = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
      mock_extract_features.return_value = (
        jnp.ones((seq_len, 10)),  # edge_features
        jnp.ones((seq_len, 48), dtype=jnp.int32),  # neighbor_indices
        jax.random.PRNGKey(42),  # prng_key
      )
      mock_project_features.return_value = jnp.ones((seq_len, 10))
      mock_encoder.return_value = (jnp.ones((seq_len, 128)), jnp.ones((seq_len, 10)))
      mock_decoder.return_value = jnp.ones((seq_len, 128))
      mock_final_projection.return_value = jnp.ones((seq_len, 20))

      # Create inputs for this sequence length
      prng_key = jax.random.PRNGKey(42)
      sequence = jnp.ones((seq_len,), dtype=jnp.int32)
      structure_coordinates = jnp.ones((seq_len, 37, 3))
      mask = jnp.ones((seq_len, 37), dtype=jnp.bool_)
      residue_index = jnp.arange(seq_len)
      chain_index = jnp.zeros((seq_len,), dtype=jnp.int32)
      k_neighbors = 48
      backbone_noise = jnp.array(0.0)
      ar_mask = None

      score, logits, decoding_order = scoring_fn(
        prng_key,
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

  def test_scoring_function_types(self, mock_model_parameters, mock_decoding_order_fn):
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
    self,
    mock_generate_ar_mask,
    mock_final_projection,
    mock_project_features,
    mock_extract_features,
    mock_make_decoder,
    mock_make_encoder,
    mock_model_parameters,
    mock_decoding_order_fn,
    sample_inputs,
  ):
    """Test the scoring output calculation logic."""
    # Setup mocks
    mock_encoder = Mock()
    mock_decoder = Mock()
    mock_make_encoder.return_value = mock_encoder
    mock_make_decoder.return_value = mock_decoder
    
    seq_len = sample_inputs["sequence"].shape[0]
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
      jax.random.PRNGKey(42),  # prng_key
    )
    mock_project_features.return_value = jnp.ones((seq_len, 10))
    mock_encoder.return_value = (jnp.ones((seq_len, 128)), jnp.ones((seq_len, 10)))
    mock_decoder.return_value = jnp.ones((seq_len, 128))

    scoring_fn = make_score_sequence(
      model_parameters=mock_model_parameters,
      decoding_order_fn=mock_decoding_order_fn,
    )

    score, logits, decoding_order = scoring_fn(
      sample_inputs["prng_key"],
      sample_inputs["sequence"],
      sample_inputs["structure_coordinates"],
      sample_inputs["mask"],
      sample_inputs["residue_index"],
      sample_inputs["chain_index"],
      sample_inputs["k_neighbors"],
      sample_inputs["backbone_noise"],
      sample_inputs["ar_mask"],
    )

    assert jnp.all(jnp.isfinite(score))
    assert jnp.all(jnp.isfinite(logits))
    assert jnp.all(jnp.isfinite(decoding_order))
