"""Tests for batching utilities."""

import pytest
from prxteinmpnn.utils.data_structures import Protein


import jax.numpy as jnp
from prxteinmpnn.utils.batching import (
  batch_and_pad_sequences,
  perform_star_alignment,
)


def test_batch_and_pad_sequences():
  """Test batch_and_pad_sequences with various input cases."""
  sequences = ["ACDE", "AC", "ACDEFGH"]
  tokens, masks = batch_and_pad_sequences(sequences)

  assert tokens.shape == (3, 7)
  assert masks.shape == (3, 7)

  # Check tokenized values
  assert (tokens[0, :4] != -1).all()  # First sequence padded to 7
  assert (tokens[0, 4:] == -1).all()
  assert (tokens[1, :2] != -1).all()  # Second sequence padded to 7
  assert (tokens[1, 2:] == -1).all()
  assert (tokens[2, :] != -1).all()  # Third sequence is the longest

  # Check masks
  assert masks[0, :4].all() and not masks[0, 4:].any()
  assert masks[1, :2].all() and not masks[1, 2:].any()
  assert masks[2, :].all()

  # Check with padding only
  sequences = ["", "", ""]
  tokens, masks = batch_and_pad_sequences(sequences)
  assert tokens.shape == (3, 0)
  assert masks.shape == (3, 0)


def test_batch_and_pad_sequences_empty():
  """Test batch_and_pad_sequences with an empty input."""
  with pytest.raises(ValueError, match="Cannot process an empty list of sequences."):
    batch_and_pad_sequences([])


def test_perform_star_alignment():
  """Test perform_star_alignment with a set of Protein objects."""
  proteins = [
    Protein(
      coordinates=jnp.zeros((4, 37, 3)),
      aatype=jnp.array([0, 1, 2, 3]),
      atom_mask=jnp.ones((4, 37)),
      residue_index=jnp.arange(4),
      chain_index=jnp.zeros(4),
      dihedrals=None,
      one_hot_sequence=jnp.eye(21)[(jnp.array([0, 1, 2, 3]),)],
    ),
    Protein(
      coordinates=jnp.zeros((2, 37, 3)),
      aatype=jnp.array([0, 1]),
      atom_mask=jnp.ones((2, 37)),
      residue_index=jnp.arange(2),
      chain_index=jnp.zeros(2),
      dihedrals=None,
      one_hot_sequence=jnp.eye(21)[(jnp.array([0, 1]),)],
    ),
  ]

  aligned_proteins = perform_star_alignment(proteins)

  assert len(aligned_proteins) == 2
  assert aligned_proteins[0].coordinates.shape[0] == aligned_proteins[1].coordinates.shape[0]
  assert aligned_proteins[0].aatype.shape[0] == aligned_proteins[1].aatype.shape[0]
  assert aligned_proteins[0].coordinates.shape[0] >= 4  # At least as long as the longest input


def test_perform_star_alignment_empty():
  """Test perform_star_alignment with an empty input."""
  aligned_proteins = perform_star_alignment([])
  assert aligned_proteins == []