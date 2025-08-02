"""Tests for the scoring module."""

from unittest.mock import Mock, patch

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import PRNGKeyArray

from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.data_structures import ModelInputs


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
    return {
      "prng_key": jax.random.PRNGKey(42),
      "sequence": jnp.ones((seq_len,), dtype=jnp.int32),
      "structure_coordinates": jnp.ones((seq_len, 4, 3)),
      "mask": jnp.ones((seq_len,), dtype=jnp.bool_),
      "residue_indices": jnp.arange(seq_len),
      "chain_indices": jnp.zeros((seq_len,), dtype=jnp.int32),
      "k_neighbors": 48,
      "augment_eps": 0.0,
    }

  @pytest.fixture
  def mock_model_inputs(self, sample_inputs):
    """Create mock ModelInputs for testing."""
    return ModelInputs(
      structure_coordinates=sample_inputs["structure_coordinates"],
      mask=sample_inputs["mask"],
      residue_index=sample_inputs["residue_indices"],
      chain_index=sample_inputs["chain_indices"],
      k_neighbors=sample_inputs["k_neighbors"],
      augment_eps=sample_inputs["augment_eps"],
    )

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
      jnp.ones((seq_len, 10)), 
      jnp.ones((seq_len, 48), dtype=jnp.int32)
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
      model_inputs=None,
    )

    # Test scoring function
    result = scoring_fn(
      sample_inputs["prng_key"],
      sample_inputs["sequence"],
      sample_inputs["structure_coordinates"],
      sample_inputs["mask"],
      sample_inputs["residue_indices"],
      sample_inputs["chain_indices"],
      sample_inputs["k_neighbors"],
      sample_inputs["augment_eps"],
    )

    # Verify result shape and type
    chex.assert_shape(result, (seq_len,))  # Should be per-position scores
    chex.assert_type(result, jnp.floating)

  @patch("prxteinmpnn.scoring.score.make_encoder")
  @patch("prxteinmpnn.scoring.score.make_decoder")
  @patch("prxteinmpnn.scoring.score.extract_features")
  @patch("prxteinmpnn.scoring.score.project_features")
  @patch("prxteinmpnn.scoring.score.final_projection")
  @patch("prxteinmpnn.scoring.score.generate_ar_mask")
  def test_make_score_sequence_with_model_inputs(
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
    mock_model_inputs,
  ):
    """Test make_score_sequence with model inputs returns partial scoring function."""
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
      jnp.ones((seq_len, 48), dtype=jnp.int32)
    )
    mock_project_features.return_value = jnp.ones((seq_len, 10))
    mock_encoder.return_value = (jnp.ones((seq_len, 128)), jnp.ones((seq_len, 10)))
    mock_decoder.return_value = jnp.ones((seq_len, 128))
    mock_final_projection.return_value = jnp.ones((seq_len, 20))

    # Create scoring function with model inputs
    scoring_fn = make_score_sequence(
      model_parameters=mock_model_parameters,
      decoding_order_fn=mock_decoding_order_fn,
      num_encoder_layers=3,
      num_decoder_layers=3,
      model_inputs=mock_model_inputs,
    )

    # Test scoring function (should only need prng_key and sequence)
    result = scoring_fn(
      sample_inputs["prng_key"],
      sample_inputs["sequence"],
    )

    # Verify result shape and type
    chex.assert_shape(result, (seq_len,))  # Should be per-position scores
    chex.assert_type(result, jnp.floating)

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
    from prxteinmpnn.model.decoder import DecodingEnum
    from prxteinmpnn.model.masked_attention import MaskedAttentionEnum

    make_score_sequence(
      model_parameters=mock_model_parameters,
      decoding_order_fn=mock_decoding_order_fn,
      num_encoder_layers=5,
      num_decoder_layers=7,
    )

    # Verify encoder creation
    mock_make_encoder.assert_called_once_with(
      model_parameters=mock_model_parameters,
      attention_mask_enum=MaskedAttentionEnum.CROSS,
      num_encoder_layers=5,
    )

    # Verify decoder creation
    mock_make_decoder.assert_called_once_with(
      model_parameters=mock_model_parameters,
      attention_mask_enum=MaskedAttentionEnum.NONE,
      decoding_enum=DecodingEnum.CONDITIONAL,
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
      jnp.ones((seq_len, 48), dtype=jnp.int32)
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
    compiled_fn = jax.jit(scoring_fn, static_argnames=("k_neighbors", "augment_eps"))
    
    # Test that compiled function works
    result = compiled_fn(
      sample_inputs["prng_key"],
      sample_inputs["sequence"],
      sample_inputs["structure_coordinates"],
      sample_inputs["mask"],
      sample_inputs["residue_indices"],
      sample_inputs["chain_indices"],
      k_neighbors=48,
      augment_eps=0.0,
    )

    chex.assert_shape(result, (seq_len,))
    chex.assert_type(result, jnp.floating)

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
      # Fix: neighbor_indices should be integer type
      mock_extract_features.return_value = (
        jnp.ones((seq_len, 10)),
        jnp.ones((seq_len, 48), dtype=jnp.int32)
      )
      mock_project_features.return_value = jnp.ones((seq_len, 10))
      mock_encoder.return_value = (jnp.ones((seq_len, 128)), jnp.ones((seq_len, 10)))
      mock_decoder.return_value = jnp.ones((seq_len, 128))
      mock_final_projection.return_value = jnp.ones((seq_len, 20))

      # Create inputs for this sequence length
      inputs = {
        "prng_key": jax.random.PRNGKey(42),
        "sequence": jnp.ones((seq_len,), dtype=jnp.int32),
        "structure_coordinates": jnp.ones((seq_len, 4, 3)),
        "mask": jnp.ones((seq_len,), dtype=jnp.bool_),
        "residue_indices": jnp.arange(seq_len),
        "chain_indices": jnp.zeros((seq_len,), dtype=jnp.int32),
        "k_neighbors": 48,
        "augment_eps": 0.0,
      }

      result = scoring_fn(**inputs)
      chex.assert_shape(result, (seq_len,))
      chex.assert_type(result, jnp.floating)

  def test_scoring_function_types(self, mock_model_parameters, mock_decoding_order_fn):
    """Test that the correct function types are returned."""
    # Test without model inputs
    scoring_fn_base = make_score_sequence(
      model_parameters=mock_model_parameters,
      decoding_order_fn=mock_decoding_order_fn,
      model_inputs=None,
    )
    
    # Should be callable with all parameters
    assert callable(scoring_fn_base)

    # Test with model inputs - create a proper mock with required attributes
    mock_model_inputs = Mock(spec=ModelInputs)
    mock_model_inputs.structure_coordinates = jnp.ones((10, 4, 3))
    mock_model_inputs.mask = jnp.ones((10,), dtype=jnp.bool_)
    mock_model_inputs.residue_index = jnp.arange(10)
    mock_model_inputs.chain_index = jnp.zeros((10,), dtype=jnp.int32)
    mock_model_inputs.k_neighbors = 48
    mock_model_inputs.augment_eps = 0.0
    
    scoring_fn_partial = make_score_sequence(
      model_parameters=mock_model_parameters,
      decoding_order_fn=mock_decoding_order_fn,
      model_inputs=mock_model_inputs,
    )
    
    # Should be callable (partial function)
    assert callable(scoring_fn_partial)

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
      jnp.ones((seq_len, 48), dtype=jnp.int32)
    )
    mock_project_features.return_value = jnp.ones((seq_len, 10))
    mock_encoder.return_value = (jnp.ones((seq_len, 128)), jnp.ones((seq_len, 10)))
    mock_decoder.return_value = jnp.ones((seq_len, 128))

    scoring_fn = make_score_sequence(
      model_parameters=mock_model_parameters,
      decoding_order_fn=mock_decoding_order_fn,
    )

    result = scoring_fn(
      sample_inputs["prng_key"],
      sample_inputs["sequence"],
      sample_inputs["structure_coordinates"],
      sample_inputs["mask"],
      sample_inputs["residue_indices"],
      sample_inputs["chain_indices"],
      sample_inputs["k_neighbors"],
      sample_inputs["augment_eps"],
    )

    # The function returns per-position scores
    expected_shape = (seq_len,)
    chex.assert_shape(result, expected_shape)
    assert jnp.all(jnp.isfinite(result))
