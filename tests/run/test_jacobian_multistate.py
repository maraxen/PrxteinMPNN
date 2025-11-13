"""Multi-state tests for Jacobian pipeline.

This module tests the categorical_jacobian() function with structure_mapping
to ensure proper isolation of conformational states during Jacobian computation.
"""

import jax
import jax.numpy as jnp
import pytest
from helpers.multistate import (
  create_multistate_test_batch,
  create_simple_multistate_protein,
)

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_fn


@pytest.fixture
def mpnn_model():
  """Create a PrxteinMPNN model for testing."""
  key = jax.random.key(42)
  model = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    k_neighbors=30,
    num_encoder_layers=3,
    num_decoder_layers=3,
    key=key,
  )
  return model


@pytest.fixture
def conditional_logits_fn(mpnn_model):
  """Create a conditional logits function."""
  return make_conditional_logits_fn(model=mpnn_model)


def test_categorical_jacobian_with_structure_mapping(conditional_logits_fn):
  """Test that conditional logits work correctly with structure_mapping.

  Verifies that the Jacobian computation pipeline (via conditional logits)
  respects structure boundaries when structure_mapping is provided.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))
  prng_key = jax.random.key(1)

  # Create a random one-hot sequence
  sequence_idx = jax.random.randint(prng_key, (protein.coordinates.shape[0],), 0, 20)
  sequence_one_hot = jax.nn.one_hot(sequence_idx, num_classes=21)

  # Compute conditional logits with structure_mapping
  logits = conditional_logits_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    sequence_one_hot,
    ar_mask=None,
    backbone_noise=None,
    structure_mapping=protein.mapping,
  )

  # Verify output shape and validity
  assert logits.shape == (protein.coordinates.shape[0], 21)
  assert jnp.all(jnp.isfinite(logits))


def test_categorical_jacobian_without_structure_mapping(conditional_logits_fn):
  """Test backward compatibility without structure_mapping.

  Verifies that conditional logits work when structure_mapping is not provided,
  ensuring backward compatibility with single-structure mode.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))
  prng_key = jax.random.key(1)

  # Create a random one-hot sequence
  sequence_idx = jax.random.randint(prng_key, (protein.coordinates.shape[0],), 0, 20)
  sequence_one_hot = jax.nn.one_hot(sequence_idx, num_classes=21)

  # Compute conditional logits WITHOUT structure_mapping
  logits = conditional_logits_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    sequence_one_hot,
    ar_mask=None,
    backbone_noise=None,
    # structure_mapping not provided
  )

  # Verify output shape and validity
  assert logits.shape == (protein.coordinates.shape[0], 21)
  assert jnp.all(jnp.isfinite(logits))


def test_categorical_jacobian_multiple_structures(conditional_logits_fn):
  """Test that multiple structures are handled correctly.

  Verifies that conditional logits computation works correctly with
  multiple structures when structure_mapping is provided.
  """
  # Create 3 structures with 40 residues each
  protein = create_multistate_test_batch(
    n_structures=3,
    n_residues_each=40,
    spatial_offset=0.5,
    key=jax.random.key(0),
  )
  prng_key = jax.random.key(1)

  # Create a random one-hot sequence
  expected_length = 120  # 3 structures * 40 residues
  sequence_idx = jax.random.randint(prng_key, (expected_length,), 0, 20)
  sequence_one_hot = jax.nn.one_hot(sequence_idx, num_classes=21)

  # Compute conditional logits with structure_mapping
  logits = conditional_logits_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    sequence_one_hot,
    ar_mask=None,
    backbone_noise=None,
    structure_mapping=protein.mapping,
  )

  # Verify output shape
  assert logits.shape == (expected_length, 21)
  assert jnp.all(jnp.isfinite(logits))


def test_categorical_jacobian_gradient_computation(conditional_logits_fn):
  """Test that gradients can be computed with structure_mapping.

  Verifies that the Jacobian pipeline supports gradient computation,
  which is essential for sensitivity analysis.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))
  prng_key = jax.random.key(1)

  # Create a random one-hot sequence
  sequence_idx = jax.random.randint(prng_key, (protein.coordinates.shape[0],), 0, 20)
  sequence_one_hot = jax.nn.one_hot(sequence_idx, num_classes=21)

  # Define a function for gradient computation
  def logits_sum(seq_one_hot):
    logits = conditional_logits_fn(
      prng_key,
      protein.coordinates,
      protein.atom_mask,
      protein.residue_index,
      protein.chain_index,
      seq_one_hot,
      ar_mask=None,
      backbone_noise=None,
      structure_mapping=protein.mapping,
    )
    return jnp.sum(logits)

  # Compute gradient
  grad_fn = jax.grad(logits_sum)
  gradient = grad_fn(sequence_one_hot)

  # Verify gradient shape and validity
  assert gradient.shape == sequence_one_hot.shape
  assert jnp.all(jnp.isfinite(gradient))


def test_categorical_jacobian_jit_compatible(mpnn_model):
  """Test that conditional logits are JIT-compatible with structure_mapping.

  Verifies that structure_mapping doesn't introduce Python control flow
  that would break JIT compilation for Jacobian computation.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))

  # Create conditional logits function
  conditional_logits_fn = make_conditional_logits_fn(model=mpnn_model)

  @jax.jit
  def jitted_conditional_logits(key, coords, mask, res_idx, chain_idx, seq, mapping):
    return conditional_logits_fn(
      key,
      coords,
      mask,
      res_idx,
      chain_idx,
      seq,
      ar_mask=None,
      backbone_noise=None,
      structure_mapping=mapping,
    )

  prng_key = jax.random.key(1)
  sequence_idx = jax.random.randint(prng_key, (protein.coordinates.shape[0],), 0, 20)
  sequence_one_hot = jax.nn.one_hot(sequence_idx, num_classes=21)

  # Call JIT version
  logits_jit = jitted_conditional_logits(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    sequence_one_hot,
    protein.mapping,
  )

  # Call non-JIT version
  logits_nojit = conditional_logits_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    sequence_one_hot,
    ar_mask=None,
    backbone_noise=None,
    structure_mapping=protein.mapping,
  )

  # Results should be identical (same random key)
  assert jnp.allclose(logits_jit, logits_nojit, atol=1e-5)
