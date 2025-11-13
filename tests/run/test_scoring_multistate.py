"""Multi-state tests for scoring pipeline.

This module tests the score_sequence() function with structure_mapping
to ensure proper isolation of conformational states during sequence scoring.
"""

import jax
import jax.numpy as jnp
import pytest
from helpers.multistate import (
  create_multistate_test_batch,
  create_simple_multistate_protein,
)

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.utils.decoding_order import random_decoding_order


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
def score_fn(mpnn_model):
  """Create a score_sequence function."""
  return make_score_sequence(
    model=mpnn_model,
    decoding_order_fn=random_decoding_order,
  )


def test_score_sequences_with_structure_mapping(score_fn):
  """Test that score_sequence works correctly with structure_mapping.

  Verifies that sequence scoring respects structure boundaries when
  structure_mapping is provided.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))
  prng_key = jax.random.key(1)

  # Create a random sequence to score
  sequence = jax.random.randint(prng_key, (protein.coordinates.shape[0],), 0, 20)

  # Score sequence with structure_mapping
  score, logits, decoding_order = score_fn(
    prng_key,
    sequence,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    structure_mapping=protein.mapping,
  )

  # Verify output shapes and validity
  assert score.shape == ()  # Scalar score
  assert logits.shape == (protein.coordinates.shape[0], 21)
  assert decoding_order.shape == (protein.coordinates.shape[0],)

  # Verify score is finite and positive (negative log-likelihood)
  assert jnp.isfinite(score)
  assert score >= 0  # NLL is non-negative

  # Verify logits are finite
  assert jnp.all(jnp.isfinite(logits))


def test_score_sequences_without_structure_mapping(score_fn):
  """Test backward compatibility without structure_mapping.

  Verifies that scoring works when structure_mapping is not provided,
  ensuring backward compatibility with single-structure mode.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))
  prng_key = jax.random.key(1)

  # Create a random sequence to score
  sequence = jax.random.randint(prng_key, (protein.coordinates.shape[0],), 0, 20)

  # Score WITHOUT structure_mapping
  score, logits, decoding_order = score_fn(
    prng_key,
    sequence,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    # structure_mapping not provided
  )

  # Verify output shapes and validity
  assert score.shape == ()
  assert logits.shape == (protein.coordinates.shape[0], 21)
  assert decoding_order.shape == (protein.coordinates.shape[0],)

  # Verify score is valid
  assert jnp.isfinite(score)
  assert score >= 0
  assert jnp.all(jnp.isfinite(logits))


def test_score_sequences_multiple_structures(score_fn):
  """Test that multiple structures are scored correctly.

  Verifies that scoring works correctly with multiple structures when
  structure_mapping is provided.
  """
  # Create 3 structures with 40 residues each
  protein = create_multistate_test_batch(
    n_structures=3,
    n_residues_each=40,
    spatial_offset=0.5,
    key=jax.random.key(0),
  )
  prng_key = jax.random.key(1)

  # Create a random sequence to score
  expected_length = 120  # 3 structures * 40 residues
  sequence = jax.random.randint(prng_key, (expected_length,), 0, 20)

  # Score sequence with structure_mapping
  score, logits, decoding_order = score_fn(
    prng_key,
    sequence,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    structure_mapping=protein.mapping,
  )

  # Verify output shapes
  assert score.shape == ()
  assert logits.shape == (expected_length, 21)
  assert decoding_order.shape == (expected_length,)

  # Verify score is valid
  assert jnp.isfinite(score)
  assert score >= 0
  assert jnp.all(jnp.isfinite(logits))


def test_score_sequences_one_hot_input(score_fn):
  """Test that scoring works with one-hot encoded sequences.

  Verifies that structure_mapping works when sequences are provided
  as one-hot encoded arrays instead of integer indices.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))
  prng_key = jax.random.key(1)

  # Create a random sequence as one-hot
  sequence_idx = jax.random.randint(prng_key, (protein.coordinates.shape[0],), 0, 20)
  sequence_one_hot = jax.nn.one_hot(sequence_idx, num_classes=21)

  # Score one-hot sequence with structure_mapping
  score, logits, decoding_order = score_fn(
    prng_key,
    sequence_one_hot,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    structure_mapping=protein.mapping,
  )

  # Verify output
  assert score.shape == ()
  assert jnp.isfinite(score)
  assert score >= 0
  assert logits.shape == (protein.coordinates.shape[0], 21)
  assert jnp.all(jnp.isfinite(logits))


def test_score_sequences_jit_compatible(mpnn_model):
  """Test that score_sequence is JIT-compatible with structure_mapping.

  Verifies that structure_mapping doesn't introduce Python control flow
  that would break JIT compilation.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))

  # Create score function
  score_fn = make_score_sequence(
    model=mpnn_model,
    decoding_order_fn=random_decoding_order,
  )

  @jax.jit
  def jitted_score(key, seq, coords, mask, res_idx, chain_idx, mapping):
    return score_fn(
      key,
      seq,
      coords,
      mask,
      res_idx,
      chain_idx,
      structure_mapping=mapping,
    )

  prng_key = jax.random.key(1)
  sequence = jax.random.randint(prng_key, (protein.coordinates.shape[0],), 0, 20)

  # Call JIT version
  score_jit, logits_jit, order_jit = jitted_score(
    prng_key,
    sequence,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    protein.mapping,
  )

  # Call non-JIT version
  score_nojit, logits_nojit, order_nojit = score_fn(
    prng_key,
    sequence,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    structure_mapping=protein.mapping,
  )

  # Results should be identical (same random key)
  assert jnp.allclose(score_jit, score_nojit, atol=1e-5)
  assert jnp.allclose(logits_jit, logits_nojit, atol=1e-5)
  assert jnp.array_equal(order_jit, order_nojit)
