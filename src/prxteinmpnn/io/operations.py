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


def pad_and_collate_proteins(elements: Sequence[ProteinTuple]) -> Protein:
  """Batch and pad a list of ProteinTuples into a ProteinBatch.

  Take a list of individual `ProteinTuple`s and batch them together into a
  single `ProteinBatch`, padding them to the maximum length in the batch.

  Args:
    elements (list[ProteinTuple]): List of protein tuples to collate.

  Returns:
    ProteinEnsemble: Batched and padded protein ensemble.

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
