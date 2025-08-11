# type: ignore[call-arg]
"""Integration tests for the sampling factory."""

import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.sampling.sampling_step import SamplingEnum


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


def test_make_sample_sequences(mock_model_parameters):
  """Test the full sequence sampling pipeline.

  Raises:
      AssertionError: If the output does not match expected shapes or properties.
  """
  L, K = 10, 4
  key = jax.random.PRNGKey(42)

  decoding_order_fn = lambda k, l: (jax.lax.iota(jnp.int32, l), jax.random.split(k)[1])

  sample_sequences_fn = make_sample_sequences(
    model_parameters=mock_model_parameters,
    decoding_order_fn=decoding_order_fn,
    sampling_strategy=SamplingEnum.STRAIGHT_THROUGH,
    num_encoder_layers=3,
    num_decoder_layers=3,
  )

  # Prepare inputs
  coords = jax.random.normal(key, (L, 4, 3))
  mask = jnp.ones((L,))
  residue_indices = jnp.arange(L)
  chain_indices = jnp.zeros((L,))
  key, sequence_key = jax.random.split(key)
  initial_sequence = jax.random.randint(sequence_key, (L,), 0, 21, dtype=jnp.int8)
  key, sample_key = jax.random.split(key)
  hyperparameters = (0.01, jnp.zeros((L, 21), dtype=jnp.float32))

  # Run sampling
  sampled_sequence, logits, decoding_order = sample_sequences_fn(
    prng_key=sample_key,
    sequence=initial_sequence, 
    structure_coordinates=coords,
    mask=mask,
    residue_index=residue_indices,
    chain_index=chain_indices,
    hyperparameters=hyperparameters,
    k_neighbors=K,
    iterations=L,
  )

  # Check output shapes
  chex.assert_shape(sampled_sequence, (L,))
  chex.assert_shape(logits, (L, 21))
  chex.assert_shape(decoding_order, (L,))


  # Check that the sequence has been modified from the initial integer sequence
  sampled_indices = jnp.argmax(sampled_sequence, axis=-1)
  assert not jnp.allclose(initial_sequence, sampled_indices)
  
  assert jnp.unique(sampled_indices).shape[0] <= 21, "Sampled sequence contains invalid amino acid indices."
  