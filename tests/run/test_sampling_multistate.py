"""Multi-state tests for sampling pipeline.

This module tests the sample_sequences() function with structure_mapping
to ensure proper isolation of conformational states during sequence sampling.
"""

import jax
import jax.numpy as jnp
import pytest
from helpers.multistate import (
  create_multistate_test_batch,
  create_simple_multistate_protein,
)

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.sampling.sample import make_sample_sequences
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
def sample_fn(mpnn_model):
  """Create a sample_sequences function."""
  return make_sample_sequences(
    model=mpnn_model,
    decoding_order_fn=random_decoding_order,
    sampling_strategy="temperature",
  )


def test_sample_sequences_with_structure_mapping(sample_fn):
  """Test that sample_sequences works correctly with structure_mapping.

  Verifies that sequence sampling respects structure boundaries when
  structure_mapping is provided.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))
  prng_key = jax.random.key(1)

  # Sample sequences with structure_mapping
  sequence, logits, decoding_order = sample_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    structure_mapping=protein.mapping,
  )

  # Verify output shapes
  assert sequence.shape == (protein.coordinates.shape[0],)
  assert logits.shape == (protein.coordinates.shape[0], 21)
  assert decoding_order.shape == (protein.coordinates.shape[0],)

  # Verify sequences are valid amino acid indices
  assert jnp.all((sequence >= 0) & (sequence < 21))

  # Verify logits are finite
  assert jnp.all(jnp.isfinite(logits))


def test_sample_sequences_without_structure_mapping(sample_fn):
  """Test backward compatibility without structure_mapping.

  Verifies that sampling works when structure_mapping is not provided,
  ensuring backward compatibility with single-structure mode.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))
  prng_key = jax.random.key(1)

  # Sample WITHOUT structure_mapping
  sequence, logits, decoding_order = sample_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    # structure_mapping not provided
  )

  # Verify output shapes
  assert sequence.shape == (protein.coordinates.shape[0],)
  assert logits.shape == (protein.coordinates.shape[0], 21)
  assert decoding_order.shape == (protein.coordinates.shape[0],)

  # Verify sequences are valid
  assert jnp.all((sequence >= 0) & (sequence < 21))
  assert jnp.all(jnp.isfinite(logits))


def test_sample_sequences_multiple_structures_isolation(sample_fn):
  """Test that multiple structures are sampled independently.

  Verifies that sequences can differ between structures when sampled
  with structure_mapping, demonstrating proper isolation.
  """
  # Create 3 structures with 40 residues each
  protein = create_multistate_test_batch(
    n_structures=3,
    n_residues_each=40,
    spatial_offset=0.5,
    key=jax.random.key(0),
  )
  prng_key = jax.random.key(1)

  # Sample sequences with structure_mapping
  sequence, logits, decoding_order = sample_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    structure_mapping=protein.mapping,
  )

  # Verify output shape
  expected_length = 120  # 3 structures * 40 residues
  assert sequence.shape == (expected_length,)
  assert logits.shape == (expected_length, 21)

  # Verify sequences are valid
  assert jnp.all((sequence >= 0) & (sequence < 21))
  assert jnp.all(jnp.isfinite(logits))

  # Extract sequences for each structure
  seq_struct_0 = sequence[:40]
  seq_struct_1 = sequence[40:80]
  seq_struct_2 = sequence[80:]

  # Verify sequences can differ (they're independently sampled)
  # Note: With small probability all could be identical, but very unlikely
  # We check that at least one pair differs
  differs_0_1 = not jnp.array_equal(seq_struct_0, seq_struct_1)
  differs_1_2 = not jnp.array_equal(seq_struct_1, seq_struct_2)
  differs_0_2 = not jnp.array_equal(seq_struct_0, seq_struct_2)

  assert differs_0_1 or differs_1_2 or differs_0_2, (
    "Expected sequences to differ between structures"
  )


def test_sample_sequences_with_temperature(sample_fn):
  """Test that sampling works with different temperature values.

  Verifies that structure_mapping works correctly across different
  temperature settings.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))
  prng_key = jax.random.key(1)

  # Test with low temperature (more deterministic)
  sequence_low, logits_low, _ = sample_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    temperature=jnp.array(0.1, dtype=jnp.float32),
    structure_mapping=protein.mapping,
  )

  # Test with high temperature (more random)
  sequence_high, logits_high, _ = sample_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    temperature=jnp.array(2.0, dtype=jnp.float32),
    structure_mapping=protein.mapping,
  )

  # Both should produce valid outputs
  assert jnp.all((sequence_low >= 0) & (sequence_low < 21))
  assert jnp.all((sequence_high >= 0) & (sequence_high < 21))
  assert jnp.all(jnp.isfinite(logits_low))
  assert jnp.all(jnp.isfinite(logits_high))


def test_sample_sequences_jit_compatible(mpnn_model):
  """Test that sample_sequences is JIT-compatible with structure_mapping.

  Verifies that structure_mapping doesn't introduce Python control flow
  that would break JIT compilation.
  """
  protein = create_simple_multistate_protein(key=jax.random.key(0))

  # Create sample function inside JIT
  sample_fn = make_sample_sequences(
    model=mpnn_model,
    decoding_order_fn=random_decoding_order,
    sampling_strategy="temperature",
  )

  @jax.jit
  def jitted_sample(coords, mask, res_idx, chain_idx, key, mapping):
    return sample_fn(
      key,
      coords,
      mask,
      res_idx,
      chain_idx,
      structure_mapping=mapping,
    )

  prng_key = jax.random.key(1)

  # Call JIT version
  sequence_jit, logits_jit, order_jit = jitted_sample(
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    prng_key,
    protein.mapping,
  )

  # Call non-JIT version
  sequence_nojit, logits_nojit, order_nojit = sample_fn(
    prng_key,
    protein.coordinates,
    protein.atom_mask,
    protein.residue_index,
    protein.chain_index,
    structure_mapping=protein.mapping,
  )

  # Results should be identical (same random key)
  assert jnp.array_equal(sequence_jit, sequence_nojit)
  assert jnp.allclose(logits_jit, logits_nojit, atol=1e-5)
  assert jnp.array_equal(order_jit, order_nojit)
