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
  and assign sequential chain IDs based on structure boundaries.

  Args:
    elements: List of protein tuples to concatenate.

  Returns:
    Protein: Single concatenated protein with remapped chain IDs.

  Raises:
    ValueError: If the input list is empty.

  Example:
    >>> combined = concatenate_proteins_for_inter_mode([protein1, protein2])
    >>> # Chain IDs will be [0,0,0..., 1,1,1..., 2,2,2...] for each structure

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

  # Get lengths of each structure
  lengths = [p.coordinates.shape[0] for p in proteins]

  # Generate sequential chain IDs: [0,0,0..., 1,1,1..., 2,2,2...]
  chain_ids = jnp.concatenate(
    [jnp.full(length, i, dtype=jnp.int32) for i, length in enumerate(lengths)],
  )

  # Concatenate all fields along residue dimension
  concatenated = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), *proteins)

  # Override chain_index with our sequential IDs
  concatenated = concatenated.replace(chain_index=chain_ids)

  # Add batch dimension of 1 to match expected shape (batch_size=1, seq_length, ...)
  return jax.tree_util.tree_map(lambda x: x[None, ...], concatenated)


def pad_and_collate_proteins(elements: Sequence[ProteinTuple]) -> Protein:
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
    padded_p = jax.tree_util.tree_map(
      lambda x, pad_len=pad_len: jnp.pad(x, ((0, pad_len),) + ((0, 0),) * (x.ndim - 1)),
      p,
    )
    padded_proteins.append(padded_p)

  return jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *padded_proteins)
