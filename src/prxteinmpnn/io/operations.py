"""Data operations for processing protein structures within a Grain pipeline.

This module implements `grain.transforms.Map` and `grain.IterOperation` classes
for parsing, transforming, and batching protein data.
"""

import warnings
from collections.abc import Sequence

import jax
import jax.numpy as jnp

from prxteinmpnn.physics.features import compute_electrostatic_features_batch
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


def _validate_and_flatten_elements(
  elements: Sequence[ProteinTuple],
) -> list[ProteinTuple]:
  """Ensure all elements are ProteinTuple and flatten nested sequences.

  Args:
    elements (Sequence[ProteinTuple]): List of protein tuples to validate.

  Returns:
    list[ProteinTuple]: Validated and flattened list of ProteinTuple.

  Raises:
    ValueError: If the input list is empty or too deeply nested.

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
  return list(elements)


def _apply_electrostatics_if_needed(
  elements: list[ProteinTuple],
  *,
  use_electrostatics: bool,
) -> list[ProteinTuple]:
  """Apply electrostatic features if requested.

  Args:
    elements (list[ProteinTuple]): List of protein tuples.
    use_electrostatics (bool): Whether to compute and add electrostatic features.

  Returns:
    list[ProteinTuple]: Updated list with electrostatic features if requested.

  """
  if not use_electrostatics:
    return elements
  phys_feats, _ = compute_electrostatic_features_batch(elements)
  return [p._replace(physics_features=feat) for p, feat in zip(elements, phys_feats, strict=False)]


def _pad_protein(protein: Protein, max_len: int) -> Protein:
  """Pad a single Protein to max_len.

  Args:
    protein (Protein): Protein to pad.
    max_len (int): Maximum length to pad to.

  Returns:
    Protein: Padded protein.

  """
  pad_len = max_len - protein.coordinates.shape[0]
  protein_len = protein.coordinates.shape[0]
  full_coords_len = (
    protein.full_coordinates.shape[0] if protein.full_coordinates is not None else None
  )
  full_coords_pad_len = max_len - full_coords_len if full_coords_len is not None else 0

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
    if hasattr(x, "__array__"):
      x = jnp.asarray(x)
    if x.ndim == 0:
      return x

    if full_coords_len is not None and x.shape[0] == full_coords_len:
      return jnp.pad(x, ((0, full_coords_pad_len),) + ((0, 0),) * (x.ndim - 1))

    if x.shape[0] == protein_len:
      return jnp.pad(x, ((0, pad_len),) + ((0, 0),) * (x.ndim - 1))

    return x

  return jax.tree_util.tree_map(pad_fn, protein)


def _stack_padded_proteins(
  padded_proteins: list[Protein],
) -> Protein:
  """Stack a list of padded Proteins into a batch.

  Args:
    padded_proteins (list[Protein]): List of padded proteins.

  Returns:
    Protein: Batched protein.

  """

  def stack_fn(*arrays: jnp.ndarray | None) -> jnp.ndarray | None:
    """Stack arrays, handling None values and scalars."""
    non_none = [a for a in arrays if a is not None]
    if not non_none:
      return None
    first = non_none[0]
    if not hasattr(first, "shape") or first.ndim == 0:
      return first
    if not all(hasattr(a, "shape") and a.shape == first.shape for a in non_none):
      return None
    return jnp.stack(non_none, axis=0)

  return jax.tree_util.tree_map(stack_fn, *padded_proteins)


def pad_and_collate_proteins(
  elements: Sequence[ProteinTuple],
  *,
  use_electrostatics: bool = False,
  _use_vdw: bool = False,
) -> Protein:
  """Batch and pad a list of ProteinTuples into a ProteinBatch.

  Take a list of individual `ProteinTuple`s and batch them together into a
  single `ProteinBatch`, padding them to the maximum length in the batch.

  Args:
    elements (list[ProteinTuple]): List of protein tuples to collate.
    use_electrostatics (bool): Whether to compute and add electrostatic features.

  Returns:
    Protein: Batched and padded protein ensemble.

  Raises:
    ValueError: If the input list is empty.

  Example:
    >>> ensemble = pad_and_collate_proteins([protein_tuple1, protein_tuple2],
    use_electrostatics=True)

  """
  elements = _validate_and_flatten_elements(elements)
  elements = _apply_electrostatics_if_needed(elements, use_electrostatics=use_electrostatics)
  proteins = [Protein.from_tuple(p) for p in elements]
  max_len = max(p.coordinates.shape[0] for p in proteins)
  padded_proteins = [_pad_protein(p, max_len) for p in proteins]
  return _stack_padded_proteins(padded_proteins)
