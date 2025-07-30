"""Unit tests for the encoder module in PrxteinMPNN.

These tests cover the core encoder functions, including parameter tree creation,
encoding, normalization, and the encoder factory functions.

All tests use mock data and minimal model parameters to ensure deterministic and
isolated behavior.

Run with:
  pytest src/prxteinmpnn/model/test_encoder.py
"""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.encoder import (
  encode,
  encoder_normalize,
  encoder_parameter_pytree,
  initialize_node_features,
  make_encode_layer,
  make_encoder,
  setup_encoder,
)
from prxteinmpnn.model.masked_attention import MaskedAttentionEnum

# --- Fixtures and helpers ---


@pytest.fixture
def minimal_model_parameters():
  """Fixture for minimal model parameters for encoder layers."""
  # Shapes: (in_dim, out_dim) or (out_dim,)
  # For simplicity, use small dims.
  params = {}
  for i in range(3):
    prefix = "protein_mpnn/~/enc_layer"
    if i > 0:
      prefix += f"_{i}"
    layer_name_suffix = f"enc{i}"
    params[f"{prefix}/~/{layer_name_suffix}_W1"] = {"w": jnp.ones((8, 8)), "b": jnp.zeros((8,))}
    params[f"{prefix}/~/{layer_name_suffix}_W2"] = {"w": jnp.ones((8, 8)), "b": jnp.zeros((8,))}
    params[f"{prefix}/~/{layer_name_suffix}_W3"] = {"w": jnp.ones((8, 8)), "b": jnp.zeros((8,))}
    params[f"{prefix}/~/{layer_name_suffix}_norm1"] = {
      "scale": jnp.ones((8,)),
      "offset": jnp.zeros((8,)),
    }
    params[f"{prefix}/~/position_wise_feed_forward/~/{layer_name_suffix}_dense_W_in"] = {
      "w": jnp.ones((8, 8)),
      "b": jnp.zeros((8,)),
    }
    params[f"{prefix}/~/position_wise_feed_forward/~/{layer_name_suffix}_dense_W_out"] = {
      "w": jnp.ones((8, 8)),
      "b": jnp.zeros((8,)),
    }
    params[f"{prefix}/~/{layer_name_suffix}_norm2"] = {
      "scale": jnp.ones((8,)),
      "offset": jnp.zeros((8,)),
    }
    params[f"{prefix}/~/{layer_name_suffix}_W11"] = {"w": jnp.ones((16, 8)), "b": jnp.zeros((8,))}
    params[f"{prefix}/~/{layer_name_suffix}_W12"] = {"w": jnp.ones((8, 8)), "b": jnp.zeros((8,))}
    params[f"{prefix}/~/{layer_name_suffix}_W13"] = {"w": jnp.ones((8, 8)), "b": jnp.zeros((8,))}
    params[f"{prefix}/~/{layer_name_suffix}_norm3"] = {
      "scale": jnp.ones((16,)),
      "offset": jnp.zeros((16,)),
    }
  # For node feature initialization
  params["protein_mpnn/~/W_e"] = {"b": jnp.zeros((8,))}
  params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), params)
  return params


@pytest.fixture
def dummy_inputs():
  """Fixture for dummy node, edge features, neighbor indices, and masks."""
  num_atoms = 4
  num_neighbors = 3
  node_dim = 8
  edge_dim = 8
  # Node features: (num_atoms, node_dim)
  node_features = jnp.ones((num_atoms, node_dim))
  # Edge features: (num_atoms, num_neighbors, edge_dim)
  edge_features = jnp.ones((num_atoms, num_neighbors, edge_dim))
  # Neighbor indices: (num_atoms, num_neighbors)
  neighbor_indices = (
    jnp.arange(num_atoms * num_neighbors).reshape(num_atoms, num_neighbors) % num_atoms
  )
  # Atom mask: (num_atoms,)
  mask = jnp.ones((num_atoms,), dtype=jnp.float32)
  # Attention mask: (num_atoms, num_neighbors)
  attention_mask = jnp.ones((num_atoms, num_neighbors), dtype=jnp.float32)
  return node_features, edge_features, neighbor_indices, mask, attention_mask


# --- Tests ---


def test_encoder_parameter_pytree_shape(minimal_model_parameters):
  """Test that encoder_parameter_pytree returns correct shapes and stacking."""
  pytree = encoder_parameter_pytree(minimal_model_parameters, num_encoder_layers=3)
  # Should be a dict of arrays stacked along axis 0 (layers)
  assert isinstance(pytree, dict)
  for v in jax.tree_util.tree_leaves(pytree):
    assert v.shape[0] == 3, "Layer param should be stacked for 3 layers"


def test_initialize_node_features_shape(minimal_model_parameters, dummy_inputs):
  """Test initialize_node_features returns correct shape."""
  _, edge_features, *_ = dummy_inputs
  node_features = initialize_node_features(minimal_model_parameters, edge_features)
  assert node_features.shape == (
    edge_features.shape[0],
    minimal_model_parameters["protein_mpnn/~/W_e"]["b"].shape[0],
  )


def test_encode_output_shape(dummy_inputs, minimal_model_parameters):
  """Test encode returns correct message shape."""
  node_features, edge_features, neighbor_indices, *_ = dummy_inputs
  # Use first layer params
  enc_pytree = encoder_parameter_pytree(minimal_model_parameters, num_encoder_layers=3)
  layer_params = jax.tree_util.tree_map(lambda x: x[0], enc_pytree)
  message = encode(node_features, edge_features, neighbor_indices, layer_params)
  assert message.shape == (node_features.shape[0], edge_features.shape[1], 8)


def test_encoder_normalize_shapes(dummy_inputs, minimal_model_parameters):
  """Test encoder_normalize returns correct node and edge feature shapes."""
  node_features, edge_features, neighbor_indices, mask, _ = dummy_inputs
  enc_pytree = encoder_parameter_pytree(minimal_model_parameters, num_encoder_layers=3)
  layer_params = jax.tree_util.tree_map(lambda x: x[0], enc_pytree)
  message = encode(node_features, edge_features, neighbor_indices, layer_params)
  node_out, edge_out = encoder_normalize(
    message,
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    layer_params,
    scale=30.0,
  )
  assert node_out.shape == node_features.shape
  assert edge_out.shape[0] == edge_features.shape[0]
  assert edge_out.shape[1] == edge_features.shape[1]


@pytest.mark.parametrize(
  "attention_mask_enum",
  [MaskedAttentionEnum.NONE, MaskedAttentionEnum.CROSS, MaskedAttentionEnum.CONDITIONAL],
)
def test_make_encode_layer(dummy_inputs, minimal_model_parameters, attention_mask_enum):
  """Test make_encode_layer returns a callable and runs without error."""
  node_features, edge_features, neighbor_indices, mask, attention_mask = dummy_inputs
  enc_pytree = encoder_parameter_pytree(minimal_model_parameters, num_encoder_layers=3)
  layer_params = jax.tree_util.tree_map(lambda x: x[0], enc_pytree)
  encode_layer_fn = make_encode_layer(attention_mask_enum)
  if attention_mask_enum is MaskedAttentionEnum.NONE:
    node_out, edge_out = encode_layer_fn(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      layer_params,
      30.0,
    )
  else:
    node_out, edge_out = encode_layer_fn(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      attention_mask,
      layer_params,
      30.0,
    )
  assert node_out.shape == node_features.shape
  assert edge_out.shape[0] == edge_features.shape[0]


def test_setup_encoder_returns_pytree_and_fn(minimal_model_parameters):
  """Test setup_encoder returns encoder params and a callable."""
  params, fn = setup_encoder(
    minimal_model_parameters,
    MaskedAttentionEnum.NONE,
    num_encoder_layers=3,
  )
  assert isinstance(params, dict)
  assert callable(fn)


@pytest.mark.parametrize(
  "attention_mask_enum",
  [MaskedAttentionEnum.NONE, MaskedAttentionEnum.CROSS, MaskedAttentionEnum.CONDITIONAL],
)
def test_make_encoder_runs(dummy_inputs, minimal_model_parameters, attention_mask_enum):
  """Test make_encoder returns a callable that runs and outputs correct shapes."""
  _, edge_features, neighbor_indices, mask, attention_mask = dummy_inputs
  encoder_fn = make_encoder(
    minimal_model_parameters,
    attention_mask_enum,
    num_encoder_layers=3,
    scale=30.0,
  )
  if attention_mask_enum is MaskedAttentionEnum.NONE:
    node_out, edge_out = encoder_fn(edge_features, neighbor_indices, mask)
  else:
    node_out, edge_out = encoder_fn(edge_features, neighbor_indices, mask, attention_mask)
  assert node_out.shape[0] == edge_features.shape[0]
  assert edge_out.shape[0] == edge_features.shape[0]
