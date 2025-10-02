"""Tests for the STE sequence optimization module."""

import jax
import jax.numpy as jnp
import pytest
from jax import random
from jaxtyping import PRNGKeyArray

from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  EdgeFeatures,
  ModelParameters,
  NeighborIndices,
  NodeFeatures,
)


@pytest.fixture
def mock_data(mock_model_parameters):
  """Create mock data for testing."""
  key = random.PRNGKey(0)
  num_residues = 5
  num_features = 128

  node_features = random.normal(key, (num_residues, num_features))
  edge_features = random.normal(key, (num_residues, 32, num_features))
  neighbor_indices = jnp.zeros((num_residues, 32), dtype=jnp.int32)
  mask = jnp.ones((num_residues,), dtype=jnp.float32)

  return {
    'node_features': node_features,
    'edge_features': edge_features,
    'neighbor_indices': neighbor_indices,
    'mask': mask,
    'model_parameters': mock_model_parameters,
    'num_residues': num_residues,
  }


def mock_conditional_decoder(
  node_features: NodeFeatures,
  edge_features: EdgeFeatures,
  neighbor_indices: NeighborIndices,
  mask: AlphaCarbonMask,
  autoregressive_mask: jax.Array,
  sequence_one_hot: jax.Array,
) -> jax.Array:
  """Mock conditional decoder that returns node features unchanged."""
  return node_features


def mock_decoding_order_fn(key: PRNGKeyArray, num_residues: int) -> tuple[jax.Array, PRNGKeyArray]:
  """Mock decoding order function that returns sequential ordering."""
  new_key = random.split(key)[0]
  return jnp.arange(num_residues), new_key


def test_optimize_sequence(mock_data):
  """Test that sequence optimization runs without errors."""
  optimize_fn = make_optimize_sequence_fn(
    mock_conditional_decoder,
    mock_decoding_order_fn,
    mock_data['model_parameters'],
  )

  key = random.PRNGKey(0)
  sequence, logits = optimize_fn(
    key,
    mock_data['node_features'],
    mock_data['edge_features'],
    mock_data['neighbor_indices'],
    mock_data['mask'],
    2,
    0.001,
    0.1,
  )

  assert isinstance(sequence, jnp.ndarray)
  assert sequence.shape == (mock_data['num_residues'], 21)
  assert sequence.dtype == jnp.float32
  assert isinstance(logits, jnp.ndarray)
  assert logits.shape == (mock_data['num_residues'], 21)


def test_optimize_sequence_differentiable(mock_data):
  """Test that the optimization process is differentiable."""
  optimize_fn = make_optimize_sequence_fn(
    mock_conditional_decoder,
    mock_decoding_order_fn,
    mock_data['model_parameters'],
  )

  def loss_wrapper(params: ModelParameters):
    sequence, _ = optimize_fn(
      random.PRNGKey(0),
      mock_data['node_features'],
      mock_data['edge_features'],
      mock_data['neighbor_indices'],
      mock_data['mask'],
      1,
      0.001,
      0.1,
    )
    return jnp.sum(sequence)

  # Test that we can take gradients
  grad_fn = jax.grad(loss_wrapper)
  grads = grad_fn(mock_data['model_parameters'])
  
  # Verify gradients for specific model components
  for layer_name, layer_params in grads.items():
    if isinstance(layer_params, dict):
      for param_name, param_value in layer_params.items():
        assert not jnp.all(jnp.isnan(param_value)), f"NaN gradients in {layer_name}/{param_name}"
  assert not jnp.all(jnp.isnan(grads['protein_mpnn/~/W_out']['w']))
  assert not jnp.all(jnp.isnan(grads['protein_mpnn/~/W_out']['b']))
