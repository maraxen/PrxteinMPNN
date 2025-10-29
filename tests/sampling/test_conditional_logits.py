"""Integration tests for the conditional logits functionality."""

import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.conditional_logits import (
    make_conditional_logits_fn,
    make_encoding_conditional_logits_split_fn,
)


@pytest.fixture
def mock_model_parameters():
  """Create a complete and structurally correct set of mock model parameters."""
  # Model dimensions
  C_V = 128  # Node feature dimension
  C_E = 128  # Edge feature dimension
  INITIAL_EDGE_FEATURES = 528
  ENCODER_MLP_INPUT_DIM = C_V + C_V + C_E  # 384
  # CORRECTED: Define the decoder's specific input dimension
  DECODER_MLP_INPUT_DIM = C_V + ENCODER_MLP_INPUT_DIM # 128 + 384 = 512
  NUM_AMINO_ACIDS = 21
  MAXIMUM_RELATIVE_FEATURES = 32
  pos_enc_dim = 2 * MAXIMUM_RELATIVE_FEATURES + 2

  # Helper to create mock linear layer parameters
  def _make_linear_params(d_in, d_out):
    return {"w": jax.random.normal(jax.random.PRNGKey(0), (d_in, d_out)), "b": jax.random.normal(jax.random.PRNGKey(0), (d_out,))}

  # Helper to create mock norm layer parameters
  def _make_norm_params(dim=C_V):
    return {"scale": jnp.ones((dim,)), "offset": jnp.zeros((dim,))}

  params = {
    # Feature extraction parameters
    "protein_mpnn/~/protein_features/~/positional_encodings/~/embedding_linear": _make_linear_params(
      pos_enc_dim, C_E
    ),
    "protein_mpnn/~/protein_features/~/edge_embedding": _make_linear_params(
      INITIAL_EDGE_FEATURES, C_E
    ),
    "protein_mpnn/~/protein_features/~/norm_edges": _make_norm_params(dim=C_E),
    # Main model parameters
    "protein_mpnn/~/W_e": _make_linear_params(C_E, C_E),
    "protein_mpnn/~/embed_token": {"W_s": jnp.ones((NUM_AMINO_ACIDS, C_V))},
    "protein_mpnn/~/W_out": _make_linear_params(C_V, NUM_AMINO_ACIDS),
  }

  # Encoder and Decoder Layers
  for i in range(3):
    enc_l_name = f"enc{i}"
    dec_l_name = f"dec{i}"
    enc_prefix = f"protein_mpnn/~/enc_layer_{i}" if i > 0 else "protein_mpnn/~/enc_layer"
    dec_prefix = f"protein_mpnn/~/dec_layer_{i}" if i > 0 else "protein_mpnn/~/dec_layer"

    # Encoder
    params.update(
      {
        f"{enc_prefix}/~/{enc_l_name}_norm1": _make_norm_params(),
        f"{enc_prefix}/~/{enc_l_name}_norm2": _make_norm_params(),
        f"{enc_prefix}/~/{enc_l_name}_norm3": _make_norm_params(),
        f"{enc_prefix}/~/{enc_l_name}_W1": _make_linear_params(ENCODER_MLP_INPUT_DIM, C_V),
        f"{enc_prefix}/~/{enc_l_name}_W2": _make_linear_params(C_V, C_V),
        f"{enc_prefix}/~/{enc_l_name}_W3": _make_linear_params(C_V, C_V),
        f"{enc_prefix}/~/{enc_l_name}_W11": _make_linear_params(ENCODER_MLP_INPUT_DIM, C_E),
        f"{enc_prefix}/~/{enc_l_name}_W12": _make_linear_params(C_E, C_E),
        f"{enc_prefix}/~/{enc_l_name}_W13": _make_linear_params(C_E, C_E),
        f"{enc_prefix}/~/position_wise_feed_forward/~/{enc_l_name}_dense_W_in": _make_linear_params(
          C_V, C_V
        ),
        f"{enc_prefix}/~/position_wise_feed_forward/~/{enc_l_name}_dense_W_out": _make_linear_params(
          C_V, C_V
        ),
      }
    )
    # Decoder
    params.update(
      {
        f"{dec_prefix}/~/{dec_l_name}_norm1": _make_norm_params(),
        f"{dec_prefix}/~/{dec_l_name}_norm2": _make_norm_params(),
        # CORRECTED: The decoder's W1 layer expects a 512-dim input.
        f"{dec_prefix}/~/{dec_l_name}_W1": _make_linear_params(DECODER_MLP_INPUT_DIM, C_V),
        f"{dec_prefix}/~/{dec_l_name}_W2": _make_linear_params(C_V, C_V),
        f"{dec_prefix}/~/{dec_l_name}_W3": _make_linear_params(C_V, C_V),
        f"{dec_prefix}/~/position_wise_feed_forward/~/{dec_l_name}_dense_W_in": _make_linear_params(
          C_V, C_V
        ),
        f"{dec_prefix}/~/position_wise_feed_forward/~/{dec_l_name}_dense_W_out": _make_linear_params(
          C_V, C_V
        ),
      }
    )

  return params


def test_make_conditional_logits_fn(mock_model_parameters):
  """Test the conditional logits function creation and execution.

  Args:
    mock_model_parameters: Mock model parameters fixture.

  Raises:
    AssertionError: If the output does not match expected shapes or properties.

  Example:
    >>> test_make_conditional_logits_fn(mock_model_parameters)
  """
  L, K = 10, 4
  key = jax.random.PRNGKey(42)

  decoding_order_fn = lambda k, l: (jax.lax.iota(jnp.int32, l), jax.random.split(k)[1])

  conditional_logits_fn = make_conditional_logits_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=decoding_order_fn,
    num_encoder_layers=3,
    num_decoder_layers=3,
  )

  # Prepare inputs
  coords = jax.random.normal(key, (L, 4, 3))
  mask = jnp.ones((L,))
  residue_indices = jnp.arange(L)
  chain_indices = jnp.zeros((L,))
  key, sequence_key = jax.random.split(key)
  sequence = jax.random.randint(sequence_key, (L,), 0, 21, dtype=jnp.int32)
  key, logits_key = jax.random.split(key)

  # Run conditional logits
  logits, node_features, edge_features = conditional_logits_fn(
    prng_key=logits_key,
    structure_coordinates=coords,
    sequence=sequence,
    mask=mask,
    residue_index=residue_indices,
    chain_index=chain_indices,
    k_neighbors=K,
  )

  # Check output shape
  chex.assert_shape(logits, (L, 21))
  chex.assert_shape(node_features, (L, 128))
  chex.assert_shape(edge_features, (L, 4, 128))

  # Check that logits are finite
  assert jnp.all(jnp.isfinite(logits)), "Logits contain non-finite values"
  assert jnp.all(jnp.isfinite(node_features)), "Node features contain non-finite values"
  assert jnp.all(jnp.isfinite(edge_features)), "Edge features contain non-finite values"

  # Check that logits have reasonable range
  assert jnp.all(logits > -100) and jnp.all(logits < 100), "Logits are outside reasonable range"


def test_conditional_logits_with_bias(mock_model_parameters):
  """Test the conditional logits function with input bias.

  Args:
    mock_model_parameters: Mock model parameters fixture.

  Raises:
    AssertionError: If the output does not match expected behavior with bias.

  Example:
    >>> test_conditional_logits_with_bias(mock_model_parameters)
  """
  L, K = 5, 4
  key = jax.random.PRNGKey(123)

  decoding_order_fn = lambda k, l: (jax.lax.iota(jnp.int32, l), jax.random.split(k)[1])

  conditional_logits_fn = make_conditional_logits_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=decoding_order_fn,
    num_encoder_layers=3,
    num_decoder_layers=3,
  )

  # Prepare inputs
  coords = jax.random.normal(key, (L, 4, 3))
  mask = jnp.ones((L,))
  residue_indices = jnp.arange(L)
  chain_indices = jnp.zeros((L,))
  key, sequence_key = jax.random.split(key)
  sequence = jax.random.randint(sequence_key, (L,), 0, 21, dtype=jnp.int32)
  key, logits_key, bias_key = jax.random.split(key, 3)

  # Create bias
  bias = jax.random.normal(bias_key, (L, 21))

  # Run conditional logits without bias
  logits_no_bias, _, _ = conditional_logits_fn(
    prng_key=logits_key,
    structure_coordinates=coords,
    sequence=sequence,
    mask=mask,
    residue_index=residue_indices,
    chain_index=chain_indices,
    k_neighbors=K,
  )

  # Run conditional logits with bias
  logits_with_bias, _, _ = conditional_logits_fn(
    prng_key=logits_key,
    structure_coordinates=coords,
    sequence=sequence,
    mask=mask,
    residue_index=residue_indices,
    chain_index=chain_indices,
    bias=bias,
    k_neighbors=K,
  )

  # Check that bias is correctly applied
  expected_logits = logits_no_bias + bias
  chex.assert_trees_all_close(logits_with_bias, expected_logits, rtol=1e-6)


def test_conditional_logits_deterministic(mock_model_parameters):
  """Test that conditional logits are deterministic for the same inputs.

  Args:
    mock_model_parameters: Mock model parameters fixture.

  Raises:
    AssertionError: If the outputs are not deterministic.

  Example:
    >>> test_conditional_logits_deterministic(mock_model_parameters)
  """
  L, K = 8, 4
  key = jax.random.PRNGKey(456)

  decoding_order_fn = lambda k, l: (jax.lax.iota(jnp.int32, l), jax.random.split(k)[1])

  conditional_logits_fn = make_conditional_logits_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=decoding_order_fn,
    num_encoder_layers=3,
    num_decoder_layers=3,
  )

  # Prepare inputs
  coords = jax.random.normal(key, (L, 4, 3))
  mask = jnp.ones((L,))
  residue_indices = jnp.arange(L)
  chain_indices = jnp.zeros((L,))
  key, sequence_key = jax.random.split(key)
  sequence = jax.random.randint(sequence_key, (L,), 0, 21, dtype=jnp.int32)
  key, logits_key = jax.random.split(key)

  # Run conditional logits twice with the same key
  logits1 = conditional_logits_fn(
    prng_key=logits_key,
    structure_coordinates=coords,
    sequence=sequence,
    mask=mask,
    residue_index=residue_indices,
    chain_index=chain_indices,
    k_neighbors=K,
    backbone_noise=0.0,  # No augmentation for determinism
  )

  logits2 = conditional_logits_fn(
    prng_key=logits_key,
    structure_coordinates=coords,
    sequence=sequence,
    mask=mask,
    residue_index=residue_indices,
    chain_index=chain_indices,
    k_neighbors=K,
    backbone_noise=0.0,  # No augmentation for determinism
  )

  # Check that outputs are identical
  chex.assert_trees_all_close(logits1, logits2, rtol=1e-10)


def test_make_encoding_conditional_logits_split_fn(mock_model_parameters):
  """Test the split encoding and conditional logits functions.

  Args:
    mock_model_parameters: Mock model parameters fixture.

  Raises:
    AssertionError: If the output does not match expected shapes or properties.

  Example:
    >>> test_make_encoding_conditional_logits_split_fn(mock_model_parameters)
  """
  L, K = 10, 4
  key = jax.random.PRNGKey(42)

  decoding_order_fn = lambda k, l: (jax.lax.iota(jnp.int32, l), jax.random.split(k)[1])

  encode_fn, condition_logits_fn = make_encoding_conditional_logits_split_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=decoding_order_fn,
    num_encoder_layers=3,
    num_decoder_layers=3,
  )

  # Prepare inputs
  coords = jax.random.normal(key, (L, 4, 3))
  mask = jnp.ones((L,))
  residue_indices = jnp.arange(L)
  chain_indices = jnp.zeros((L,))
  key, sequence_key = jax.random.split(key)
  sequence = jax.random.randint(sequence_key, (L,), 0, 21, dtype=jnp.int32)
  key, encode_key = jax.random.split(key)

  # Run encode function
  (
    node_features,
    edge_features,
    neighbor_indices,
    mask_out,
    autoregressive_mask,
  ) = encode_fn(
    prng_key=encode_key,
    structure_coordinates=coords,
    mask=mask,
    residue_index=residue_indices,
    chain_index=chain_indices,
    k_neighbors=K,
  )

  # Check encode outputs
  chex.assert_shape(node_features, (L, 128))
  chex.assert_shape(edge_features, (L, K, 128))
  chex.assert_shape(neighbor_indices, (L, K))
  chex.assert_shape(mask_out, (L,))
  chex.assert_shape(autoregressive_mask, (L, L))

  # Run condition_logits function
  logits, decoded_node_features, _ = condition_logits_fn(
    node_features=node_features,
    edge_features=edge_features,
    neighbor_indices=neighbor_indices,
    mask=mask_out,
    autoregressive_mask=autoregressive_mask,
    sequence=sequence,
  )

  # Check condition_logits outputs
  chex.assert_shape(logits, (L, 21))
  chex.assert_shape(decoded_node_features, (L, 128))
  assert jnp.all(jnp.isfinite(logits)), "Logits contain non-finite values"
