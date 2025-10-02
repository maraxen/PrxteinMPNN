"""Provides a high-level API for creating Grain-based data loaders."""

import pathlib
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

import grain

from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

from . import operations, sources
from .cache import preprocess_inputs_to_hdf5


def create_protein_dataset(
  inputs: str | Path | Sequence[str | Path | IO[str]],
  batch_size: int,
  num_workers: int = 0,
  parse_kwargs: dict[str, Any] | None = None,
  foldcomp_database: FoldCompDatabase | None = None,
  cache_path: str | pathlib.Path | None = None,
) -> grain.IterDataset:
  """Construct a high-performance protein data pipeline using Grain.

  This function sets up a data loading pipeline that preprocesses inputs,
  caches them to an HDF5 file, and then efficiently reads protein structure
  frames.

  Args:
      inputs: A single input (file, PDB ID, etc.) or a sequence of such inputs.
      batch_size: The number of protein structures to include in each batch.
      num_workers: The number of parallel worker processes for data loading.
                   0 means all loading is done in the main process.
      parse_kwargs: Optional dictionary of keyword arguments for parsing.
      foldcomp_database: Optional path to a FoldComp database.
      cache_path: Optional path to cache the preprocessed HDF5 file. If None,
                  a default path is used.

  Returns:
      A Grain IterDataset that yields batches of padded `Protein` objects.

  """
  if not isinstance(inputs, Sequence) or isinstance(inputs, (str, pathlib.Path)):
    inputs = (inputs,)

  parse_kwargs = parse_kwargs or {}

  hdf5_path = preprocess_inputs_to_hdf5(
    inputs,
    output_path=cache_path,
    parse_kwargs=parse_kwargs,
    foldcomp_database=foldcomp_database,
  )
  source = sources.HDF5DataSource(hdf5_path)

  ds = grain.MapDataset.source(source)
  ds = ds.map(operations.LoadHDF5Frame(hdf5_path))

  ds = ds.to_iter_dataset()
  ds = ds.batch(batch_size, batch_fn=operations.pad_and_collate_proteins)

  if num_workers > 0:
    mp_options = grain.MultiprocessingOptions(num_workers=num_workers)
    ds = ds.mp_prefetch(mp_options)

  return ds