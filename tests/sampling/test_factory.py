"""Integration tests for the sampling factory."""

import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.sampling.factory import make_sample_sequences
from prxteinmpnn.utils.data_structures import SamplingEnum


@pytest.fixture
def mock_model_parameters():
  """Create a simplified but structurally correct set of model parameters."""
  C_V, C_E = 128, 128
  # This structure needs to be more detailed to match what the model expects.
  params = {
    "embedding": {
      "node_features": jnp.ones((21, C_V)),
      "edge_features": jnp.ones((34, C_E)),
    },
    "final_projection": {"kernel": jnp.ones((C_V, 21))},
  }

  # Add encoder and decoder layer parameters
  for i in range(3):  # Assuming max 3 layers for the test
    enc_prefix = f"protein_mpnn/~/enc_layer_{i}" if i > 0 else "protein_mpnn/~/enc_layer"
    dec_prefix = f"protein_mpnn/~/dec_layer_{i}" if i > 0 else "protein_mpnn/~/dec_layer"
    enc_l_name = f"enc{i}"
    dec_l_name = f"dec{i}"
    params.update(
      {
        f"{enc_prefix}/~/{enc_l_name}_W1": jnp.ones((C_V, C_V)),
        f"{enc_prefix}/~/{enc_l_name}_W2": jnp.ones((C_V, C_V)),
        f"{enc_prefix}/~/{enc_l_name}_W3": jnp.ones((C_V, C_V)),
        f"{enc_prefix}/~/{enc_l_name}_W11": jnp.ones((C_E, C_E)),
        f"{enc_prefix}/~/{enc_l_name}_W12": jnp.ones((C_E, C_E)),
        f"{enc_prefix}/~/{enc_l_name}_W13": jnp.ones((C_E, C_E)),
        f"{enc_prefix}/~/{enc_l_name}_norm1": {"scale": 1.0, "offset": 0.0},
        f"{enc_prefix}/~/{enc_l_name}_norm2": {"scale": 1.0, "offset": 0.0},
        f"{enc_prefix}/~/{enc_l_name}_norm3": {"scale": 1.0, "offset": 0.0},
        f"{enc_prefix}/~/position_wise_feed_forward/~/{enc_l_name}_dense_W_in": jnp.ones(
          (C_V, C_V)
        ),
        f"{enc_prefix}/~/position_wise_feed_forward/~/{enc_l_name}_dense_W_out": jnp.ones(
          (C_V, C_V)
        ),
        f"{dec_prefix}/~/{dec_l_name}_W1": jnp.ones((C_V, C_V)),
        f"{dec_prefix}/~/{dec_l_name}_W2": jnp.ones((C_V, C_V)),
        f"{dec_prefix}/~/{dec_l_name}_W3": jnp.ones((C_V, C_V)),
        f"{dec_prefix}/~/{dec_l_name}_W11": jnp.ones((C_E, C_E)),
        f"{dec_prefix}/~/{dec_l_name}_W12": jnp.ones((C_E, C_E)),
        f"{dec_prefix}/~/{dec_l_name}_W13": jnp.ones((C_E, C_E)),
        f"{dec_prefix}/~/{dec_l_name}_norm1": {"scale": 1.0, "offset": 0.0},
        f"{dec_prefix}/~/{dec_l_name}_norm2": {"scale": 1.0, "offset": 0.0},
        f"{dec_prefix}/~/{dec_l_name}_norm3": {"scale": 1.0, "offset": 0.0},
        f"{dec_prefix}/~/position_wise_feed_forward/~/{dec_l_name}_dense_W_in": jnp.ones(
          (C_V, C_V)
        ),
        f"{dec_prefix}/~/position_wise_feed_forward/~/{dec_l_name}_dense_W_out": jnp.ones(
          (C_V, C_V)
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

  # A simple decoding order function that returns indices in order
  decoding_order_fn = lambda k, l: (jnp.arange(l), jax.random.split(k)[1])

  # Create the sampling function
  sample_sequences_fn = make_sample_sequences(
    model_parameters=mock_model_parameters,
    decoding_order_fn=decoding_order_fn,
    sampling_strategy=SamplingEnum.TEMPERATURE,
    num_encoder_layers=1,
    num_decoder_layers=1,
  )

  # Prepare inputs
  coords = jax.random.normal(key, (L, 4, 3))
  mask = jnp.ones((L,))
  residue_indices = jnp.arange(L)
  chain_indices = jnp.zeros((L,))
  initial_sequence = jax.nn.one_hot(jnp.zeros(L, dtype=jnp.int32), 21)
  key, sample_key = jax.random.split(key)

  # Run sampling
  sampled_sequence, logits, decoding_order = sample_sequences_fn(
    prng_key=sample_key,
    initial_sequence=initial_sequence,
    structure_coordinates=coords,
    mask=mask,
    residue_indices=residue_indices,
    chain_indices=chain_indices,
    k_neighbors=K,
    iterations=L,
  )

  # Check output shapes
  chex.assert_shape(sampled_sequence, (L, 21))
  chex.assert_shape(logits, (L, 21))
  chex.assert_shape(decoding_order, (L,))

  # Check sequence properties
  chex.assert_trees_all_close(jnp.sum(sampled_sequence, axis=-1), jnp.ones(L))

  # Check that the sequence has been modified from the initial all-zero sequence
  assert not jnp.allclose(initial_sequence, sampled_sequence)