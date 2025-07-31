"""Tests for the sampling initialization pass."""

import chex
import jax
import jax.numpy as jnp
from prxteinmpnn.sampling.initialize import sampling_encode


def test_sampling_encode():
  """Test the sampling encoder pass.

  Raises:
      AssertionError: If output shapes or types are incorrect.
  """
  L, K = 10, 8
  C_V, C_E = 128, 128
  key = jax.random.PRNGKey(0)

  # Mock functions and data
  mock_encoder = lambda edge_features, *_: (
    jnp.ones((L, C_V)),
    edge_features,
  )
  mock_decoding_order_fn = lambda k, l: (jnp.arange(0, l), jax.random.split(k)[1])

  sample_model_pass = sampling_encode(mock_encoder, mock_decoding_order_fn)

  coords = jnp.zeros((L, 4, 3))
  mask = jnp.ones((L,))
  res_indices = jnp.arange(L)
  chain_indices = jnp.zeros((L,))
  mock_params = {
    "embedding": {
      "node_features": jnp.ones((21, C_V)),
      "edge_features": jnp.ones((34, C_E)),
    }
  }

  (
    node_features,
    edge_features,
    neighbor_indices,
    decoding_order,
    ar_mask,
    next_key,
  ) = sample_model_pass(
    key, coords, mask, res_indices, chain_indices, mock_params, k_neighbors=K
  )

  # Assert output shapes
  chex.assert_shape(node_features, (L, C_V))
  chex.assert_shape(edge_features, (L, K, C_E))
  chex.assert_shape(neighbor_indices, (L, K))
  chex.assert_shape(decoding_order, (L,))
  chex.assert_shape(ar_mask, (L, L))
  chex.assert_shape(next_key, key.shape)