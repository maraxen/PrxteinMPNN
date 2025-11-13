"""Data operations for processing protein structures within a Grain pipeline.

This module implements `grain.transforms.Map` and `grain.IterOperation` classes
for parsing, transforming, and batching protein data.
"""

import warnings
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from prxteinmpnn.utils.data_structures import Protein, ProteinTuple

_MAX_TRIES = 5


def concatenate_proteins_for_inter_mode(elements: Sequence[ProteinTuple]) -> Protein:
  """Concatenate proteins for inter-chain mode (pass_mode='inter').

  Instead of padding and stacking, concatenate all structures along the residue dimension
  and remap chain IDs to ensure global uniqueness across all structures.

  Each structure's chain IDs are offset by the maximum chain ID from all previous structures,
  preserving the original chain relationships within each structure while ensuring no
  collisions across structures.

  The structure boundaries are stored in the `mapping` field as [0,0,0..., 1,1,1..., 2,2,2...]
  to enable "direct" tied_positions mode.

  Args:
    elements: List of protein tuples to concatenate.

  Returns:
    Protein: Single concatenated protein with globally unique chain IDs and structure mapping.

  Raises:
    ValueError: If the input list is empty.

  Example:
    >>> # Structure 1: chains [0,0,1,1], Structure 2: chains [0,0,2,2]
    >>> combined = concatenate_proteins_for_inter_mode([protein1, protein2])
    >>> # Result chains: [0,0,1,1,2,2,4,4] - each structure's chains are offset
    >>> # Result mapping: [0,0,0,0,1,1,1,1] - tracks which structure each residue came from

  """
  if not elements:
    msg = "Cannot concatenate an empty list of proteins."
    warnings.warn(msg, stacklevel=2)
    raise ValueError(msg)

  tries = 0
  while not all(isinstance(p, ProteinTuple) for p in elements):
    if any(isinstance(p, Sequence) for p in elements):
      elements = [p[0] if isinstance(p, Sequence) else p for p in elements]  # type: ignore[index]
      tries += 1
    if tries > _MAX_TRIES:
      msg = "Too many nested sequences in elements; cannot collate."
      warnings.warn(msg, stacklevel=2)
      raise ValueError(msg)

  proteins = [Protein.from_tuple(p) for p in elements]

  structure_indices = []
  for i, protein in enumerate(proteins):
    length = protein.coordinates.shape[0]
    structure_indices.append(jnp.full(length, i, dtype=jnp.int32))

  structure_mapping = jnp.concatenate(structure_indices, axis=0)
  remapped_chain_ids = []
  chain_offset = 0

  for protein in proteins:
    original_chains = protein.chain_index
    remapped_chains = original_chains + chain_offset
    remapped_chain_ids.append(remapped_chains)
    chain_offset = int(jnp.max(remapped_chains)) + 1

  chain_ids = jnp.concatenate(remapped_chain_ids, axis=0)
  concatenated = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), *proteins)
  concatenated = concatenated.replace(chain_index=chain_ids, mapping=structure_mapping)
  return jax.tree_util.tree_map(lambda x: x[None, ...], concatenated)


def pad_and_collate_proteins(elements: Sequence[ProteinTuple]) -> Protein:  # noqa: C901
  """Batch and pad a list of ProteinTuples into a ProteinBatch.

  Take a list of individual `ProteinTuple`s and batch them together into a
  single `ProteinBatch`, padding them to the maximum length in the batch.

  Args:
    elements (list[ProteinTuple]): List of protein tuples to collate.

  Returns:
    Protein: Batched and padded protein ensemble.

  Raises:
    ValueError: If the input list is empty.

  Example:
    >>> ensemble = pad_and_collate_proteins([protein_tuple1, protein_tuple2])

  """
  if not elements:
    msg = "Cannot collate an empty list of proteins."
    warnings.warn(msg, stacklevel=2)
    raise ValueError(msg)

  tries = 0
  while not all(isinstance(p, ProteinTuple) for p in elements):
    if any(isinstance(p, Sequence) for p in elements):
      elements = [p[0] if isinstance(p, Sequence) else p for p in elements]  # type: ignore[index]
      tries += 1
    if tries > _MAX_TRIES:
      msg = "Too many nested sequences in elements; cannot collate."
      warnings.warn(msg, stacklevel=2)
      raise ValueError(msg)

  proteins = [Protein.from_tuple(p) for p in elements]
  max_len = max(p.coordinates.shape[0] for p in proteins)

  padded_proteins = []
  for p in proteins:
    pad_len = max_len - p.coordinates.shape[0]
    protein_len = p.coordinates.shape[0]

    # For estat fields, we need to pad based on full_coordinates length if it exists
    full_coords_len = p.full_coordinates.shape[0] if p.full_coordinates is not None else None
    full_coords_pad_len = max_len - full_coords_len if full_coords_len is not None else 0

    # Pad function that handles None values and pads arrays based on their expected dimension
    def pad_fn(
      x: jnp.ndarray | None,
      *,
      pad_len: int = pad_len,
      protein_len: int = protein_len,
      full_coords_len: int | None = full_coords_len,
      full_coords_pad_len: int = full_coords_pad_len,
    ) -> jnp.ndarray | None:
      """Pad array along first dimension if it matches the protein residue count."""
      if x is None:
        return None
      if not hasattr(x, "shape") or not hasattr(x, "ndim"):
        return x
      # Convert numpy arrays to jax arrays
      if hasattr(x, "__array__"):
        x = jnp.asarray(x)
      # Don't pad scalars
      if x.ndim == 0:
        return x

      # Check if this matches full_coordinates length (for estat fields)
      if full_coords_len is not None and x.shape[0] == full_coords_len:
        return jnp.pad(x, ((0, full_coords_pad_len),) + ((0, 0),) * (x.ndim - 1))

      # Check if this matches coordinates length (for backbone fields)
      if x.shape[0] == protein_len:
        return jnp.pad(x, ((0, pad_len),) + ((0, 0),) * (x.ndim - 1))

      # Don't pad arrays with different lengths
      return x

    padded_p = jax.tree_util.tree_map(pad_fn, p)
    padded_proteins.append(padded_p)

  # Stack with special handling for None values and scalar arrays
  def stack_fn(*arrays: jnp.ndarray | None) -> jnp.ndarray | None:
    """Stack arrays, handling None values and scalars."""
    # Filter out None values
    non_none = [a for a in arrays if a is not None]
    if not non_none:
      return None
    # Don't stack scalars or arrays that don't match the first dimension
    first = non_none[0]
    if not hasattr(first, "shape") or first.ndim == 0:
      return first
    # Check if all arrays have the same shape
    if not all(hasattr(a, "shape") and a.shape == first.shape for a in non_none):
      # Return None for fields that can't be stacked (like estat fields with different atom counts)
      return None
    return jnp.stack(non_none, axis=0)

  return jax.tree_util.tree_map(stack_fn, *padded_proteins)
