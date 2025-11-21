"""Helper functions for multi-state protein design testing.

This module provides utilities for creating test data and verifying
correctness of multi-state implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from prxteinmpnn.utils.data_structures import Protein


def create_multistate_test_batch(
  n_structures: int,
  n_residues_each: int,
  spatial_offset: float = 0.5,
  *,
  key: jax.Array | None = None,
) -> Protein:
  """Create synthetic multi-state protein batch for testing.

  Generates concatenated protein structures with controlled spatial separation.
  Structures are positioned close enough that without structure_mapping,
  cross-structure neighbors would occur.

  Args:
    n_structures: Number of protein structures to concatenate.
    n_residues_each: Number of residues per structure.
    spatial_offset: Y-axis offset between structures (Angstroms).
                   Small values (< 1.0) ensure structures are close enough
                   to trigger cross-structure neighbors without masking.
    key: JAX random key for reproducible coordinate generation.
         If None, uses a fixed seed.

  Returns:
    Protein object with concatenated structures and mapping field.

  Example:
    >>> batch = create_multistate_test_batch(2, 3, spatial_offset=0.5)
    >>> assert batch.mapping.shape == (6,)
    >>> assert jnp.array_equal(batch.mapping, jnp.array([0,0,0,1,1,1]))

  """
  if key is None:
    key = jax.random.key(0)

  import numpy as np

  from prxteinmpnn.utils.data_structures import ProteinTuple

  total_residues = n_structures * n_residues_each

  # Generate base coordinates for one structure
  # Simple linear arrangement along x-axis
  base_coords = jnp.zeros((n_residues_each, 4, 3), dtype=jnp.float32)
  for i in range(n_residues_each):
    # N, CA, C, O atoms
    base_coords = base_coords.at[i, 0, 0].set(i * 3.8)  # N
    base_coords = base_coords.at[i, 1, 0].set(i * 3.8 + 1.5)  # CA
    base_coords = base_coords.at[i, 2, 0].set(i * 3.8 + 2.4)  # C
    base_coords = base_coords.at[i, 3, 0].set(i * 3.8 + 3.0)  # O

  # Concatenate structures with y-axis offset
  all_coords = []
  all_masks = []
  all_residue_idx = []
  all_chain_idx = []
  all_sequences = []
  structure_mapping = []

  for struct_idx in range(n_structures):
    coords = base_coords.copy()
    # Offset in y-direction
    coords = coords.at[:, :, 1].set(struct_idx * spatial_offset)

    all_coords.append(coords)
    all_masks.append(jnp.ones((n_residues_each,), dtype=jnp.float32))  # residue mask
    all_residue_idx.append(jnp.arange(n_residues_each, dtype=jnp.int32))
    all_chain_idx.append(jnp.full(n_residues_each, struct_idx, dtype=jnp.int32))
    # Random sequence for this structure
    seq_key, key = jax.random.split(key)
    all_sequences.append(jax.random.randint(seq_key, (n_residues_each,), 0, 20))
    structure_mapping.extend([struct_idx] * n_residues_each)

  coordinates = np.asarray(jnp.concatenate(all_coords, axis=0))
  atom_mask = np.asarray(jnp.concatenate(all_masks, axis=0))
  residue_index = np.asarray(jnp.concatenate(all_residue_idx, axis=0))
  chain_index = np.asarray(jnp.concatenate(all_chain_idx, axis=0))
  aatype = np.asarray(jnp.concatenate(all_sequences, axis=0))
  mapping = np.array(structure_mapping, dtype=np.int32)

  return ProteinTuple(
    coordinates=coordinates,
    aatype=aatype,
    atom_mask=atom_mask,
    residue_index=residue_index,
    chain_index=chain_index,
    mapping=mapping,
  )


def verify_no_cross_structure_neighbors(
  neighbor_indices: jax.Array,
  structure_mapping: jax.Array,
) -> tuple[bool, str]:
  """Verify that neighbors respect structure boundaries.

  Checks that for each residue, all its neighbors have the same
  structure_mapping value (i.e., are from the same structure).

  Args:
    neighbor_indices: Array of shape (N, K) where N is number of residues
                     and K is number of neighbors per residue.
    structure_mapping: Array of shape (N,) mapping each residue to a structure ID.

  Returns:
    Tuple of (is_valid, error_message) where:
      - is_valid: True if all neighbors respect structure boundaries
      - error_message: Empty string if valid, detailed error if invalid

  Example:
    >>> neighbors = jnp.array([[1, 2], [0, 2], [0, 1]])  # 3 residues, 2 neighbors
    >>> mapping = jnp.array([0, 0, 0])  # All from same structure
    >>> is_valid, msg = verify_no_cross_structure_neighbors(neighbors, mapping)
    >>> assert is_valid

  """
  n_residues, k_neighbors = neighbor_indices.shape

  violations = []

  for i in range(n_residues):
    my_structure = structure_mapping[i]
    my_neighbors = neighbor_indices[i]

    # Get structure IDs of all neighbors
    neighbor_structures = structure_mapping[my_neighbors]

    # Check if any neighbor is from a different structure
    cross_structure = neighbor_structures != my_structure

    if jnp.any(cross_structure):
      violating_neighbors = my_neighbors[cross_structure]
      violations.append(
        f"Residue {i} (structure {my_structure}) has neighbors from other structures: "
        f"{violating_neighbors.tolist()} with structure IDs {neighbor_structures[cross_structure].tolist()}",
      )

  if violations:
    error_msg = f"Found {len(violations)} violations:\n" + "\n".join(violations[:5])
    if len(violations) > 5:
      error_msg += f"\n... and {len(violations) - 5} more"
    return False, error_msg

  return True, ""


def assert_sequences_tied(
  sequences: jax.Array,
  tie_group_map: jax.Array,
) -> None:
  """Verify that sequences respect tie_group constraints.

  Checks that all positions with the same tie_group_map value have
  identical amino acids across all sequences (if multiple sequences provided).

  Args:
    sequences: Array of shape (N,) or (B, N) where B is batch size and
              N is sequence length. Contains amino acid indices.
    tie_group_map: Array of shape (N,) mapping each position to a tie group.

  Raises:
    AssertionError: If tied positions have different amino acids.

  Example:
    >>> seq = jnp.array([5, 10, 15, 5, 10, 15])  # Two copies of same pattern
    >>> tie_map = jnp.array([0, 1, 2, 0, 1, 2])  # Positions 0 and 3 tied, etc.
    >>> assert_sequences_tied(seq, tie_map)  # Passes

  """
  if sequences.ndim == 1:
    # Single sequence
    sequences = sequences[None, :]  # Add batch dimension

  batch_size, seq_length = sequences.shape

  # Get unique tie groups
  unique_groups = jnp.unique(tie_group_map)

  for group_id in unique_groups:
    # Find all positions in this tie group
    positions = jnp.where(tie_group_map == group_id)[0]

    if len(positions) <= 1:
      # No tying needed for single position
      continue

    # Check that all positions in this group have the same AA across all sequences
    for batch_idx in range(batch_size):
      group_values = sequences[batch_idx, positions]
      first_value = group_values[0]

      if not jnp.all(group_values == first_value):
        msg = (
          f"Tie group {group_id} violation in sequence {batch_idx}: "
          f"positions {positions.tolist()} have values {group_values.tolist()}, "
          f"expected all to be {first_value}"
        )
        raise AssertionError(msg)


def create_simple_multistate_protein(
  *,
  key: jax.Array | None = None,
) -> Protein:
  """Create a minimal 2-structure test protein for multi-state testing.

  Convenience function for quick testing. Creates structures close enough
  to trigger cross-structure neighbors without masking. Each structure
  has enough residues (50) to be realistic for k_neighbors=30.

  Args:
    key: JAX random key for reproducible generation.

  Returns:
    Protein with 100 total residues (50 per structure).

  Example:
    >>> protein = create_simple_multistate_protein()
    >>> assert protein.coordinates.shape == (100, 4, 3)
    >>> assert protein.mapping.shape == (100,)

  """
  return create_multistate_test_batch(
    n_structures=2,
    n_residues_each=50,  # Enough for k_neighbors=30
    spatial_offset=0.5,
    key=key,
  )
