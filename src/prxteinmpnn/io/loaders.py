"""Provides a high-level API for creating Grain-based data loaders."""

from collections.abc import Sequence
from io import StringIO
from typing import Any

import grain.python as grain

from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

from . import operations, sources


def create_protein_dataset(
  inputs: Sequence[str | StringIO],
  batch_size: int,
  foldcomp_database: FoldCompDatabase | None = None,
  parse_kwargs: dict[str, Any] | None = None,
  num_workers: int = 0,
) -> grain.IterDataset:
  """Construct a high-performance protein data pipeline using Grain.

  Args:
      inputs: A sequence of inputs (file paths, PDB IDs, StringIO, etc.).
      batch_size: The number of protein structures to include in each batch.
      foldcomp_database: An optional FoldCompDatabase for resolving FoldComp IDs.
      parse_kwargs: Optional dictionary of keyword arguments passed to the parser.
      num_workers: The number of parallel worker processes for data loading.
                   0 means all loading is done in the main process.

  Returns:
      A Grain IterDataset that yields batches of ProteinEnsemble objects.

  """
  if parse_kwargs is None:
    parse_kwargs = {}

  source = sources.MixedInputDataSource(inputs, foldcomp_database)
  ds = grain.MapDataset.source(source)

  ds = ds.map(operations.ParseStructure(parse_kwargs=parse_kwargs))

  ds = ds.filter(lambda x: x is not None)

  ds = ds.batch(batch_size, batch_fn=operations.pad_and_collate_proteins)

  iter_ds = ds.to_iter_dataset()

  if num_workers > 0:
    mp_options = grain.MultiprocessingOptions(num_workers=num_workers)
    iter_ds = iter_ds.mp_prefetch(mp_options)

  return iter_ds
