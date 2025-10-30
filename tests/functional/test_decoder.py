"""Test suite for the decoder module in the PrxteinMPNN model.

Tests the functionality of the decoder, including initialization,
decoding, normalization, and setup functions.
"""

import os
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.functional import (
  decode_message,
  decoder_normalize,
  decoder_parameter_pytree,
  extract_features,
  initialize_conditional_decoder,
  make_decode_layer,
  make_decoder,
  make_encoder,
  setup_decoder,
)
from prxteinmpnn.functional.decoder import DecodingApproach
from prxteinmpnn.model.masked_attention import MaskedAttentionType
from prxteinmpnn.utils.residue_constants import atom_order


# ruff : noqa: D102, ANN201, S101

class TestDecoderParameterPytree:
  """Test decoder parameter pytree function."""

  def test_decoder_parameter_pytree_single_layer(self):
    # Mock model parameters
    mock_params = {}
    for i in range(1):
      prefix = "protein_mpnn/~/dec_layer"
      if i > 0:
        prefix += f"_{i}"
      layer_suffix = f"dec{i}"

      mock_params.update(
        {
          f"{prefix}/~/{layer_suffix}_W1": jnp.array([[1.0, 2.0]]),
          f"{prefix}/~/{layer_suffix}_W2": jnp.array([[3.0, 4.0]]),
          f"{prefix}/~/{layer_suffix}_W3": jnp.array([[5.0, 6.0]]),
          f"{prefix}/~/{layer_suffix}_norm1": {
            "scale": jnp.array([1.0]),
            "offset": jnp.array([0.0]),
          },
          f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_in": jnp.array(
            [[7.0, 8.0]],
          ),
          f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_out": jnp.array(
            [[9.0, 10.0]],
          ),
          f"{prefix}/~/{layer_suffix}_norm2": {
            "scale": jnp.array([1.0]),
            "offset": jnp.array([0.0]),
          },
        },
      )

    result = decoder_parameter_pytree(mock_params, num_decoder_layers=1)

    assert "W1" in result
    assert "W2" in result
    assert "W3" in result
    assert "norm1" in result
    assert "dense_W_in" in result
    assert "dense_W_out" in result
    assert "norm2" in result


NUM_ATOMS = 3
NUM_NEIGHBORS = 2


class TestInitializeConditionalDecoder:
  """Test conditional decoder initialization."""

  def test_initialize_conditional_decoder(self):
    # Create mock inputs
    sequence = jax.nn.one_hot(jnp.array([1, 2, 3]), num_classes=20)  # Mock one-hot encoded sequence
    node_features = jnp.ones((3, 4))
    edge_features = jnp.ones((3, 2, 5))
    neighbor_indices = jnp.array([[0, 1], [1, 2], [0, 2]])

    # Mock layer params with embed_token
    layer_params = {
      "protein_mpnn/~/embed_token": {
        "W_s": jnp.ones((20, 4)),  # vocab_size=10, embedding_dim=4
      },
    }
    
    layer_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), layer_params)

    node_edge_features, sequence_edge_features = initialize_conditional_decoder(
      sequence,
      node_features,
      edge_features,
      neighbor_indices,
      layer_params,
    )

    assert node_edge_features.shape[0] == NUM_ATOMS  # num_atoms
    assert sequence_edge_features.shape[0] == NUM_ATOMS  # num_atoms
    assert node_edge_features.shape[1] == NUM_NEIGHBORS  # num_neighbors
    assert sequence_edge_features.shape[1] == NUM_NEIGHBORS  # num_neighbors


class TestDecode:
  """Test the decode function."""

  def test_decode_shape(self):
    node_features = jnp.ones((3, 4))
    edge_features = jnp.ones((3, 2, 5))

    # Mock layer params
    layer_params = {
      "W1": {"w": jnp.ones((9, 8)), "b": jnp.zeros(8)},  # 4+5=9 input features
      "W2": {"w": jnp.ones((8, 6)), "b": jnp.zeros(6)},
      "W3": {"w": jnp.ones((6, 4)), "b": jnp.zeros(4)},
    }
    
    layer_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), layer_params)

    result = decode_message(node_features, edge_features, layer_params)

    assert result.shape == (3, 2, 4)  # (num_atoms, num_neighbors, output_features)


class TestDecoderNormalize:
  """Test decoder normalization function."""

  def test_decoder_normalize(self):
    message = jnp.ones((3, 2, 4))
    node_features = jnp.ones((3, 4))
    mask = jnp.ones(3)

    # Mock layer params
    layer_params = {
      "norm1": {"scale": jnp.ones(4), "offset": jnp.zeros(4)},
      "norm2": {"scale": jnp.ones(4), "offset": jnp.zeros(4)},
      "dense_W_in": {"w": jnp.ones((4, 8)), "b": jnp.zeros(8)},
      "dense_W_out": {"w": jnp.ones((8, 4)), "b": jnp.zeros(4)},
    }
    

    result = decoder_normalize(message, node_features, mask, layer_params)

    assert result.shape == (3, 4)


class TestMakeDecodeLayer:
  """Test make_decode_layer function."""

  def test_make_decode_layer_cross_attention(self):
    decode_fn = make_decode_layer("cross")
    assert callable(decode_fn)

  def test_make_decode_layer_conditional_attention(self):
    decode_fn = make_decode_layer("conditional")
    assert callable(decode_fn)


class TestSetupDecoder:
  """Test setup_decoder function."""

  def test_setup_decoder(self):
    # Mock minimal model parameters
    model_params = {}
    for i in range(1):
      prefix = "protein_mpnn/~/dec_layer"
      layer_suffix = f"dec{i}"

      model_params.update(
        {
          f"{prefix}/~/{layer_suffix}_W1": jnp.array([[1.0]]),
          f"{prefix}/~/{layer_suffix}_W2": jnp.array([[1.0]]),
          f"{prefix}/~/{layer_suffix}_W3": jnp.array([[1.0]]),
          f"{prefix}/~/{layer_suffix}_norm1": {
            "scale": jnp.array([1.0]),
            "offset": jnp.array([0.0]),
          },
          f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_in": jnp.array([[1.0]]),
          f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_out": jnp.array([[1.0]]),
          f"{prefix}/~/{layer_suffix}_norm2": {
            "scale": jnp.array([1.0]),
            "offset": jnp.array([0.0]),
          },
        },
      )

    params, decode_fn = setup_decoder(
      model_params,
      None,
      "unconditional",
      num_decoder_layers=1,
    )

    assert params is not None
    assert callable(decode_fn)


class TestMakeDecoder:
  """Test make_decoder function."""

  def test_make_decoder_unconditional_no_mask(self):
    # Mock minimal model parameters
    model_params = {}
    for i in range(1):
      prefix = "protein_mpnn/~/dec_layer"
      layer_suffix = f"dec{i}"

      model_params.update(
        {
          f"{prefix}/~/{layer_suffix}_W1": jnp.ones((6, 4)),  # adjusted dimensions
          f"{prefix}/~/{layer_suffix}_W2": jnp.ones((4, 4)),
          f"{prefix}/~/{layer_suffix}_W3": jnp.ones((4, 3)),
          f"{prefix}/~/{layer_suffix}_norm1": {"scale": jnp.ones(3), "offset": jnp.zeros(3)},
          f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_in": jnp.ones((3, 6)),
          f"{prefix}/~/position_wise_feed_forward/~/{layer_suffix}_dense_W_out": jnp.ones((6, 3)),
          f"{prefix}/~/{layer_suffix}_norm2": {"scale": jnp.ones(3), "offset": jnp.zeros(3)},
        },
      )

    decoder_fn = make_decoder(
      model_params,
      None,
      "unconditional",
      num_decoder_layers=1,
    )

    assert callable(decoder_fn)


def test_decoder_with_golden(
    rng_key,
    model_inputs,
    mock_model_parameters,
):
    """
    Test the full decoder against a golden file to ensure determinism and correctness.
    If the golden file doesn't exist, it will be created.
    """
    golden_file = Path(__file__).parent / "golden_files" / "decoder_golden.npz"

    # Get Encoder inputs
    feature_inputs = model_inputs.copy()
    del feature_inputs["sequence"]
    edge_features, neighbor_indices, _ = extract_features(
        rng_key,
        mock_model_parameters,
        **feature_inputs,
    )
    mask = model_inputs["mask"]

    # Run Encoder to get Decoder inputs
    encoder_fn = make_encoder(
        mock_model_parameters,
        attention_mask_type=None,
        num_encoder_layers=3,
        scale=30.0,
    )
    node_features, edge_features = encoder_fn(edge_features, neighbor_indices, mask)

    # Run Decoder
    decoder_fn = make_decoder(
        mock_model_parameters,
        attention_mask_type=None,
        decoding_approach="unconditional",
        num_decoder_layers=3,
        scale=30.0,
    )
    decoder_output = decoder_fn(node_features, edge_features, mask)

    if not golden_file.exists():
        os.makedirs(golden_file.parent, exist_ok=True)
        np.savez(golden_file, decoder_output=decoder_output)
        pytest.skip(f"Golden file created at {golden_file}. Please re-run the tests.")

    # Load the golden data
    golden_data = np.load(golden_file)
    expected_decoder_output = golden_data["decoder_output"]

    # Compare the results
    chex.assert_trees_all_close(decoder_output, expected_decoder_output, atol=1e-6, rtol=1e-6)

  