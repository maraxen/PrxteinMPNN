"""Provides a high-level API for creating Grain-based data loaders."""

import pathlib
from collections.abc import Sequence
from pathlib import Path
from typing import IO

import grain

from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

from . import operations, sources
from .cache import preprocess_inputs_to_hdf5


def create_protein_dataset(
  inputs: str | Path | Sequence[str | Path | IO[str]],
  batch_size: int,
  num_workers: int = 0,
  parse_kwargs: dict | None = None,
  foldcomp_database: FoldCompDatabase | None = None,
  preprocess_path: str | pathlib.Path = "preprocessed_data.hdf5",
) -> grain.IterDataset:
  """Construct a high-performance protein data pipeline using Grain from a preprocessed HDF5 file.

  This function sets up a data loading pipeline that efficiently reads protein structure
  frames from an HDF5 file created by `prxteinmpnn.io.cache.preprocess_inputs_to_hdf5`.

  Args:
      inputs: A single input (file, PDB ID, etc.) or a sequence of such inputs.
      parse_kwargs: Optional dictionary of keyword arguments to pass to the parser.
      foldcomp_database: An optional FoldCompDatabase instance for resolving FoldComp IDs.
      preprocess_path: The path where the preprocessed HDF5 file will be stored.
                      If the file already exists, it will be reused.
      hdf5_path: The path to the preprocessed HDF5 file.
      batch_size: The number of protein structures to include in each batch.
      num_workers: The number of parallel worker processes for data loading.
                   0 means all loading is done in the main process.

  Returns:
      A Grain IterDataset that yields batches of padded `Protein` objects.

  """
  if not isinstance(inputs, Sequence) or isinstance(inputs, (str, pathlib.Path)):
    inputs = (inputs,)
  preprocess_inputs_to_hdf5(
    inputs,
    output_path=preprocess_path,
    parse_kwargs=parse_kwargs,
    foldcomp_database=foldcomp_database,
  )
  source = sources.HDF5DataSource(preprocess_path)

  ds = grain.MapDataset.source(source)
  ds = ds.map(operations.LoadHDF5Frame(preprocess_path))

  ds = ds.to_iter_dataset()
  ds = ds.batch(batch_size, batch_fn=operations.pad_and_collate_proteins)

  if num_workers > 0:
    mp_options = grain.MultiprocessingOptions(num_workers=num_workers)
    ds = ds.mp_prefetch(mp_options)

  return ds
