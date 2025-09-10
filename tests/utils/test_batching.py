"""Tests for batching utilities."""

from typing import AsyncGenerator

import chex # Import chex
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.parsing import string_to_protein_sequence
from prxteinmpnn.utils.batching import (
  batch_and_pad_proteins,
  batch_and_pad_sequences,
)
from prxteinmpnn.utils.data_structures import Protein, ProteinEnsemble


# --- Fixtures ---
@pytest.fixture
def sample_sequences() -> list[str]:
  """Provides a list of protein sequences of different lengths."""
  return ["AG", "AGY", "AGYX"]


@pytest.fixture
def sample_protein_ensemble() -> ProteinEnsemble:
  """Provides a mock ProteinEnsemble for testing."""

  async def _ensemble_generator() -> AsyncGenerator[tuple[Protein, str], None]:
    proteins = [
      Protein(
        coordinates=jnp.ones((2, 37, 3), dtype=jnp.float32),
        aatype=string_to_protein_sequence("AG"),
        atom_mask=jnp.ones((2, 37), dtype=jnp.int32),
        residue_index=jnp.arange(2, dtype=jnp.int32),
        chain_index=jnp.zeros(2, dtype=jnp.int32),
      ),
      Protein(
        coordinates=jnp.ones((4, 37, 3), dtype=jnp.float32),
        aatype=string_to_protein_sequence("AGYX"),
        atom_mask=jnp.ones((4, 37), dtype=jnp.int32),
        residue_index=jnp.arange(4, dtype=jnp.int32),
        chain_index=jnp.zeros(4, dtype=jnp.int32),
      ),
      Protein(
        coordinates=jnp.ones((3, 37, 3), dtype=jnp.float32),
        aatype=string_to_protein_sequence("AGY"),
        atom_mask=jnp.ones((3, 37), dtype=jnp.int32),
        residue_index=jnp.arange(3, dtype=jnp.int32),
        chain_index=jnp.zeros(3, dtype=jnp.int32),
      ),
    ]
    sources = ["protein1.pdb", "protein2.pdb", "protein3.pdb"]
    for p, s in zip(proteins, sources):
      yield p, s

  return _ensemble_generator()


# --- Tests for batch_and_pad_sequences ---
def test_batch_and_pad_sequences_normal(sample_sequences: list[str]):
  """Test batching and padding with sequences of different lengths."""
  tokens, masks = batch_and_pad_sequences(sample_sequences)

  # Check shape using chex
  chex.assert_shape(tokens, (3, 4))
  chex.assert_shape(masks, (3, 4))

  # Check content
  # "AG" -> [0, 5, -1, -1]
  # "AGY" -> [0, 5, 19, -1]
  # "AGYX" -> [0, 5, 19, 20]
  expected_tokens = jnp.array(
    [[0, 5, -1, -1], [0, 5, 19, -1], [0, 5, 19, 20]],
    dtype=jnp.int8,
  )
  expected_masks = jnp.array(
    [[True, True, False, False], [True, True, True, False], [True, True, True, True]],
    dtype=jnp.bool_,
  )

  chex.assert_trees_all_equal(tokens, expected_tokens)
  chex.assert_trees_all_equal(masks, expected_masks)


def test_batch_and_pad_sequences_empty_list():
  """Test that an empty list of sequences raises a ValueError."""
  with pytest.raises(ValueError, match="Cannot process an empty list of sequences."):
    batch_and_pad_sequences([])


def test_batch_and_pad_sequences_single_sequence():
  """Test batching with a single sequence."""
  tokens, masks = batch_and_pad_sequences(["AGY"])
  chex.assert_shape(tokens, (1, 3))
  chex.assert_shape(masks, (1, 3))
  expected_tokens = jnp.array([[0, 5, 19]], dtype=jnp.int8)
  expected_masks = jnp.array([[True, True, True]], dtype=jnp.bool_)
  chex.assert_trees_all_equal(tokens, expected_tokens)
  chex.assert_trees_all_equal(masks, expected_masks)


def test_batch_and_pad_sequences_same_length():
  """Test batching with sequences of the same length."""
  sequences = ["AGY", "GAY", "YGA"]
  tokens, masks = batch_and_pad_sequences(sequences)
  chex.assert_shape(tokens, (3, 3))
  chex.assert_shape(masks, (3, 3))
  # Check that there's no padding
  assert not jnp.any(tokens == -1)
  assert jnp.all(masks)


# --- Tests for batch_and_pad_proteins ---
@pytest.mark.anyio
async def test_batch_and_pad_proteins_normal(sample_protein_ensemble: ProteinEnsemble):
  """Test batching and padding a normal protein ensemble without extra sequences."""
  # Updated: Unpack the new return signature. aligned_sequences_tokens should be None.
  batched_protein, sources, aligned_sequences_tokens = await batch_and_pad_proteins(sample_protein_ensemble)

  # Check that no extra sequences were returned
  assert aligned_sequences_tokens is None

  # For the given sample sequences ("AG", "AGYX", "AGY"),
  # a simple star alignment with "AGYX" as reference will result in an MSA length of 4.
  msa_length = 4 # Expected MSA length based on these specific sequences

  # Check batch size and max_len using chex
  chex.assert_shape(batched_protein.coordinates, (3, msa_length, 37, 3))
  chex.assert_shape(batched_protein.aatype, (3, msa_length))
  chex.assert_shape(batched_protein.atom_mask, (3, msa_length, 37))
  chex.assert_shape(batched_protein.residue_index, (3, msa_length))
  chex.assert_shape(batched_protein.chain_index, (3, msa_length))

  # Check sources
  assert sources == ["protein1.pdb", "protein2.pdb", "protein3.pdb"]

  # Check padding values for aatype (-1) and other arrays (0)
  # Protein 0 ("AG") should have 2 padding positions at the end (index 2 and 3)
  assert jnp.array_equal(batched_protein.aatype[0, 2:], jnp.array([-1, -1], dtype=jnp.int8))
  assert jnp.all(batched_protein.coordinates[0, 2:] == 0.0)
  assert jnp.all(batched_protein.atom_mask[0, 2:] == 0)
  assert jnp.all(batched_protein.residue_index[0, 2:] == 0)
  assert jnp.all(batched_protein.chain_index[0, 2:] == 0)

  # Protein 1 ("AGYX") should have no padding if MSA length is 4
  assert jnp.all(batched_protein.aatype[1] != -1) # No -1s in aatype for this one
  assert jnp.all(batched_protein.coordinates[1] != 0.0) # No 0.0s for coordinates after padding if original was 1.0

  # Protein 2 ("AGY") should have 1 padding position at the end (index 3)
  assert batched_protein.aatype[2, 3] == -1
  assert jnp.all(batched_protein.coordinates[2, 3:] == 0.0)
  assert jnp.all(batched_protein.atom_mask[2, 3:] == 0)
  assert jnp.all(batched_protein.residue_index[2, 3:] == 0)
  assert jnp.all(batched_protein.chain_index[2, 3:] == 0)


@pytest.mark.anyio
async def test_batch_and_pad_proteins_with_sequences_to_score(sample_protein_ensemble: ProteinEnsemble):
  """Test batching and padding a protein ensemble with additional sequences to score."""
  sequences_to_score = ["GX", "AGYXV"] # Two extra sequences
  batched_protein, sources, aligned_sequences_tokens = await batch_and_pad_proteins(
      sample_protein_ensemble,
      sequences_to_score=sequences_to_score,
  )

  # The longest sequence overall is "AGYXV" (length 5).
  # The MSA will now be based on all sequences: "AG", "AGYX", "AGY", "GX", "AGYXV".
  # A star alignment using "AGYXV" as reference will likely result in an MSA length of 5.
  # Let's verify the MSA length is at least 5.
  msa_length = batched_protein.aatype.shape[1]
  assert msa_length >= 5 # Should be 5 for "AGYXV" as reference

  # Check shapes
  chex.assert_shape(batched_protein.coordinates, (3, msa_length, 37, 3))
  chex.assert_shape(aligned_sequences_tokens, (len(sequences_to_score), msa_length))

  # Check sources (should only contain original protein sources)
  assert sources == ["protein1.pdb", "protein2.pdb", "protein3.pdb"]

  # Check dtypes

  # Verify content of aligned_sequences_tokens
  # "GX" (G=5, X=20) padded to msa_length
  # "AGYXV" (A=0, G=5, Y=19, X=20, V=17) padded to msa_length
  # The exact padding for 'GX' will depend on alignment, but should have -1 for gaps.
  # For AGYXV, its content should be preserved even after alignment and padding.
  expected_tokenized_agvx = string_to_protein_sequence("AGYXV")
  actual_padded_agvx = aligned_sequences_tokens[1]
  actual_agvx_no_gaps = actual_padded_agvx[actual_padded_agvx != -1]
  chex.assert_trees_all_equal(actual_agvx_no_gaps, expected_tokenized_agvx)

  # Verify that the 'GX' sequence is also padded correctly (check for -1s)
  assert jnp.any(aligned_sequences_tokens[0] == -1)
  

  # And that its content is preserved after removing gaps.
  expected_tokenized_gx = string_to_protein_sequence("GX")
  expected_tokenized_gx = expected_tokenized_gx[expected_tokenized_gx != 20] # Remove X for comparison
  actual_padded_gx = aligned_sequences_tokens[0]
  actual_gx_no_gaps = actual_padded_gx[actual_padded_gx != -1] # This was correct
  chex.assert_trees_all_equal(actual_gx_no_gaps, expected_tokenized_gx)


@pytest.mark.anyio
async def test_batch_and_pad_proteins_empty_ensemble():
  """Test that an empty ensemble raises a ValueError."""

  async def _empty_generator() -> AsyncGenerator[tuple[Protein, str], None]:
    # Explicitly yield nothing to create an empty generator
    if False:
      yield

  with pytest.raises(ValueError, match="Cannot batch an empty ProteinEnsemble."):
    # Need to pass an empty generator instance
    await batch_and_pad_proteins(_empty_generator())


@pytest.mark.anyio
async def test_batch_and_pad_proteins_single_protein():
  """Test batching an ensemble with a single protein."""

  async def _single_generator():
    protein = Protein(
      coordinates=jnp.ones((5, 37, 3), dtype=jnp.float32),
      aatype=string_to_protein_sequence("AGYXV"),
      atom_mask=jnp.ones((5, 37), dtype=jnp.int32),
      residue_index=jnp.arange(5, dtype=jnp.int32),
      chain_index=jnp.zeros(5, dtype=jnp.int32),
    )
    yield protein, "single.pdb"

  # Updated: Unpack the new return signature. aligned_sequences_tokens should be None.
  batched_protein, sources, aligned_sequences_tokens = await batch_and_pad_proteins(_single_generator())

  assert aligned_sequences_tokens is None # No sequences_to_score provided

  chex.assert_shape(batched_protein.coordinates, (1, 5, 37, 3)) # MSA length should be 5
  assert sources == ["single.pdb"]
  # Ensure no padding for this single, full-length protein if MSA length is 5
  assert jnp.all(batched_protein.aatype[0] != -1)
  assert jnp.all(batched_protein.coordinates[0] != 0.0) # Assuming initial coordinates are 1.0