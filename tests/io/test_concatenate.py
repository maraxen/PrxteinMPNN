"""Tests for concatenate_proteins_for_inter_mode."""

import jax.numpy as jnp

from prxteinmpnn.io.operations import concatenate_proteins_for_inter_mode
from prxteinmpnn.utils.data_structures import Protein, ProteinTuple


def test_concatenate_preserves_unique_chains():
  """Test that concatenate_proteins_for_inter_mode preserves unique chain IDs."""
  # Create two proteins, each with 2 chains
  # Protein 1: 4 residues, chains [0,0,1,1]
  tuple1 = ProteinTuple(
    coordinates=jnp.zeros((4, 4, 3)),
    aatype=jnp.zeros(4, dtype=jnp.int32),
    atom_mask=jnp.ones((4, 4), dtype=jnp.bool_),
    residue_index=jnp.array([0, 1, 0, 1], dtype=jnp.int32),
    chain_index=jnp.array([0, 0, 1, 1], dtype=jnp.int32),
  )

  # Protein 2: 4 residues, chains [0,0,2,2]
  tuple2 = ProteinTuple(
    coordinates=jnp.zeros((4, 4, 3)),
    aatype=jnp.zeros(4, dtype=jnp.int32),
    atom_mask=jnp.ones((4, 4), dtype=jnp.bool_),
    residue_index=jnp.array([0, 1, 0, 1], dtype=jnp.int32),
    chain_index=jnp.array([0, 0, 2, 2], dtype=jnp.int32),
  )

  # Concatenate
  result = concatenate_proteins_for_inter_mode([tuple1, tuple2])

  # Check results
  assert result.coordinates.shape == (1, 8, 4, 3)  # batch_dim=1, total 8 residues
  assert result.chain_index.shape == (1, 8)  # noqa: PLR2004

  # Expected chain IDs after offset remapping:
  # Protein 1: [0,0,1,1] → [0,0,1,1] (no offset)
  # Protein 2: [0,0,2,2] → [2,2,4,4] (offset by max(1)+1=2)
  expected_chains = jnp.array([[0, 0, 1, 1, 2, 2, 4, 4]], dtype=jnp.int32)  # noqa: PLR2004
  assert (result.chain_index == expected_chains).all()

  # Check structure mapping
  assert result.mapping is not None
  expected_mapping = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=jnp.int32)
  assert (result.mapping == expected_mapping).all()

  # Verify all chain IDs are unique (no collisions)
  unique_chains = jnp.unique(result.chain_index[0])
  assert len(unique_chains) == 4  # 4 unique chains total  # noqa: PLR2004


def test_concatenate_single_chain_per_structure():
  """Test concatenation when each structure has only one chain."""
  # Protein 1: 3 residues, chain [0,0,0]
  tuple1 = ProteinTuple(
    coordinates=jnp.zeros((3, 4, 3)),
    aatype=jnp.zeros(3, dtype=jnp.int32),
    atom_mask=jnp.ones((3, 4), dtype=jnp.bool_),
    residue_index=jnp.array([0, 1, 2], dtype=jnp.int32),
    chain_index=jnp.array([0, 0, 0], dtype=jnp.int32),
  )

  # Protein 2: 3 residues, chain [0,0,0]
  tuple2 = ProteinTuple(
    coordinates=jnp.zeros((3, 4, 3)),
    aatype=jnp.zeros(3, dtype=jnp.int32),
    atom_mask=jnp.ones((3, 4), dtype=jnp.bool_),
    residue_index=jnp.array([0, 1, 2], dtype=jnp.int32),
    chain_index=jnp.array([0, 0, 0], dtype=jnp.int32),
  )

  result = concatenate_proteins_for_inter_mode([tuple1, tuple2])

  # Expected chain IDs:
  # Protein 1: [0,0,0] → [0,0,0] (no offset)
  # Protein 2: [0,0,0] → [1,1,1] (offset by max(0)+1=1)
  expected_chains = jnp.array([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32)
  assert (result.chain_index == expected_chains).all()

  # Check structure mapping
  expected_mapping = jnp.array([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32)
  assert (result.mapping == expected_mapping).all()
