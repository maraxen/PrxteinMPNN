"""Tests for unconditional logits sampling."""

import jax
import jax.numpy as jnp
import pytest
from jax import random
from jaxtyping import PRNGKeyArray

from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn
from prxteinmpnn.utils.data_structures import ModelInputs
from prxteinmpnn.utils.types import (
  AtomMask,
  ChainIndex,
  ResidueIndex,
  StructureAtomicCoordinates,
)


@pytest.fixture
def mock_structure_data():
  """Create mock structure data for testing."""
  num_residues = 5
  num_atoms = 4  # N, CA, C, O
  
  return {
    'structure_coordinates': jnp.ones((num_residues, num_atoms, 3)),
    'mask': jnp.ones((num_residues,), dtype=jnp.float32),
    'residue_index': jnp.arange(num_residues),
    'chain_index': jnp.zeros((num_residues,), dtype=jnp.int32),
  }


def mock_decoding_order_fn(key: PRNGKeyArray, num_residues: int) -> tuple[jax.Array, PRNGKeyArray]:
  """Mock decoding order function that returns sequential ordering."""
  new_key = random.split(key)[0]
  return jnp.arange(num_residues), new_key


def test_make_unconditional_logits_fn(mock_model_parameters, mock_structure_data):
  """Test that unconditional logits function can be created and run."""
  logits_fn = make_unconditional_logits_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=mock_decoding_order_fn,
  )

  key = random.PRNGKey(0)
  logits, node_features, edge_features = logits_fn(
    key,
    mock_structure_data['structure_coordinates'],
    mock_structure_data['mask'],
    mock_structure_data['residue_index'],
    mock_structure_data['chain_index'],
  )

  num_residues = mock_structure_data['structure_coordinates'].shape[0]
  assert logits.shape == (num_residues, 21)
  assert not jnp.any(jnp.isnan(logits))
  assert node_features.shape[0] == num_residues
  assert edge_features.shape[:2] == (num_residues, num_residues)


def test_make_unconditional_logits_fn_with_model_inputs(mock_model_parameters, mock_structure_data):
  """Test that unconditional logits function works with ModelInputs."""
  model_inputs = ModelInputs(
    structure_coordinates=mock_structure_data['structure_coordinates'],
    mask=mock_structure_data['mask'],
    residue_index=mock_structure_data['residue_index'],
    chain_index=mock_structure_data['chain_index'],
  )

  logits_fn = make_unconditional_logits_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=mock_decoding_order_fn,
    model_inputs=model_inputs,
  )

  key = random.PRNGKey(0)
  logits, node_features, edge_features = logits_fn(key)

  num_residues = mock_structure_data['structure_coordinates'].shape[0]
  assert logits.shape == (num_residues, 21)
  assert not jnp.any(jnp.isnan(logits))


def test_make_unconditional_logits_fn_with_bias(mock_model_parameters, mock_structure_data):
  """Test that unconditional logits function works with input bias."""
  logits_fn = make_unconditional_logits_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=mock_decoding_order_fn,
  )

  key = random.PRNGKey(0)
  num_residues = mock_structure_data['structure_coordinates'].shape[0]
  bias = jnp.ones((num_residues, 21))

  logits, _, _ = logits_fn(
    key,
    mock_structure_data['structure_coordinates'],
    mock_structure_data['mask'],
    mock_structure_data['residue_index'],
    mock_structure_data['chain_index'],
    bias=bias,
  )

  assert logits.shape == (num_residues, 21)
  assert not jnp.any(jnp.isnan(logits))


def test_make_unconditional_logits_fn_different_layers(mock_model_parameters, mock_structure_data):
  """Test that unconditional logits function works with different numbers of layers."""
  logits_fn = make_unconditional_logits_fn(
    model_parameters=mock_model_parameters,
    decoding_order_fn=mock_decoding_order_fn,
    num_encoder_layers=2,
    num_decoder_layers=1,
  )

  key = random.PRNGKey(0)
  logits, _, _ = logits_fn(
    key,
    mock_structure_data['structure_coordinates'],
    mock_structure_data['mask'],
    mock_structure_data['residue_index'],
    mock_structure_data['chain_index'],
  )

  num_residues = mock_structure_data['structure_coordinates'].shape[0]
  assert logits.shape == (num_residues, 21)
  assert not jnp.any(jnp.isnan(logits))
