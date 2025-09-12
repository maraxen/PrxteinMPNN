"""Tests for batching utilities."""

import pytest
import jax
import jax.numpy as jnp

from prxteinmpnn.utils.batching import (
  batch_and_pad_sequences,
  batch_and_pad_proteins,
  perform_star_alignment,
)
from prxteinmpnn.utils.data_structures import Protein, ProteinEnsemble


def test_batch_and_pad_sequences():
  """Test batch_and_pad_sequences with pre-tokenized JAX arrays."""
  # Create pre-tokenized sequences (integers)
  sequences = [
    jnp.array([1, 2, 3, 4]),  # ACDE
    jnp.array([1, 2]),        # AC  
    jnp.array([1, 2, 3, 4, 5, 6, 7])  # ACDEFGH
  ]
  
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


def test_batch_and_pad_sequences_empty_arrays():
  """Test batch_and_pad_sequences with empty arrays."""
  sequences = [jnp.array([]), jnp.array([]), jnp.array([])]
  tokens, masks = batch_and_pad_sequences(sequences)
  assert tokens.shape == (3, 0)
  assert masks.shape == (3, 0)


def test_batch_and_pad_sequences_empty_input():
  """Test batch_and_pad_sequences with an empty input."""
  with pytest.raises(ValueError, match="Cannot process an empty list of sequences."):
    batch_and_pad_sequences([])


def test_batch_and_pad_proteins_basic():
  """Test batch_and_pad_proteins with basic protein inputs."""
  proteins = [
    Protein(
      coordinates=jnp.zeros((4, 37, 3)),
      aatype=jnp.array([0, 1, 2, 3]),
      atom_mask=jnp.ones((4, 37)),
      residue_index=jnp.arange(4),
      chain_index=jnp.zeros(4),
      dihedrals=None,
      one_hot_sequence=jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 21),
    ),
    Protein(
      coordinates=jnp.zeros((2, 37, 3)),
      aatype=jnp.array([0, 1]),
      atom_mask=jnp.ones((2, 37)),
      residue_index=jnp.arange(2),
      chain_index=jnp.zeros(2),
      dihedrals=None,
      one_hot_sequence=jax.nn.one_hot(jnp.array([0, 1]), 21),
    ),
  ]

  ensemble, batched_sequences = batch_and_pad_proteins(proteins)

  # Check return types
  assert isinstance(ensemble, ProteinEnsemble)
  assert batched_sequences is None  # No sequences provided

  # Check shapes are consistent and padded
  assert ensemble.coordinates.shape == (2, 4, 37, 3)  # Padded to max length
  assert ensemble.aatype.shape == (2, 4)
  assert ensemble.atom_mask.shape == (2, 4, 37)
  assert ensemble.one_hot_sequence.shape == (2, 4, 21)
  
  # Check padding values
  assert (ensemble.aatype[1, 2:] == -1).all()  # Second protein padded with -1


def test_batch_and_pad_proteins_with_sequences():
  """Test batch_and_pad_proteins with sequences to score."""
  proteins = [
    Protein(
      coordinates=jnp.zeros((3, 37, 3)),
      aatype=jnp.array([0, 1, 2]),
      atom_mask=jnp.ones((3, 37)),
      residue_index=jnp.arange(3),
      chain_index=jnp.zeros(3),
      dihedrals=None,
      one_hot_sequence=jax.nn.one_hot(jnp.array([0, 1, 2]), 21),
    ),
  ]
  
  sequences = [jnp.array([1, 2, 3]), jnp.array([4, 5])]

  ensemble, batched_sequences = batch_and_pad_proteins(proteins, sequences_to_score=sequences)

  assert isinstance(ensemble, ProteinEnsemble)
  assert batched_sequences is not None
  assert batched_sequences.shape == (2, 3)  # 2 sequences, padded to length 3


def test_batch_and_pad_proteins_with_cross_diff():
  """Test batch_and_pad_proteins with cross-protein mapping enabled."""
  proteins = [
    Protein(
      coordinates=jnp.zeros((3, 37, 3)),
      aatype=jnp.array([0, 1, 2]),
      atom_mask=jnp.ones((3, 37)),
      residue_index=jnp.arange(3),
      chain_index=jnp.zeros(3),
      dihedrals=None,
      one_hot_sequence=jax.nn.one_hot(jnp.array([0, 1, 2]), 21),
    ),
    Protein(
      coordinates=jnp.zeros((2, 37, 3)),
      aatype=jnp.array([0, 1]),
      atom_mask=jnp.ones((2, 37)),
      residue_index=jnp.arange(2),
      chain_index=jnp.zeros(2),
      dihedrals=None,
      one_hot_sequence=jax.nn.one_hot(jnp.array([0, 1]), 21),
    ),
  ]

  ensemble, _ = batch_and_pad_proteins(proteins, calculate_cross_diff=True)

  assert isinstance(ensemble, ProteinEnsemble)
  assert ensemble.mapping is not None
  # For 2 proteins, we expect 1 pair in upper triangle storage
  assert ensemble.mapping.shape[0] == 1  # Number of pairs
  assert ensemble.mapping.shape[2] == 2   # [pos_i, pos_j] format


def test_batch_and_pad_proteins_empty():
  """Test batch_and_pad_proteins with empty input."""
  with pytest.raises(ValueError, match="Cannot process an empty list of proteins."):
    batch_and_pad_proteins([])


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
      one_hot_sequence=jax.nn.one_hot(jnp.array([0, 1, 2, 3]), 21),
    ),
    Protein(
      coordinates=jnp.zeros((2, 37, 3)),
      aatype=jnp.array([0, 1]),
      atom_mask=jnp.ones((2, 37)),
      residue_index=jnp.arange(2),
      chain_index=jnp.zeros(2),
      dihedrals=None,
      one_hot_sequence=jax.nn.one_hot(jnp.array([0, 1]), 21),
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