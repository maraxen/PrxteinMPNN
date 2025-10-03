"""Provides a high-level API for creating Grain-based data loaders."""

import pathlib
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

import grain

from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

from . import operations, sources
from .cache import preprocess_inputs_to_hdf5
from .prefetch_autotune import pick_performance_config


def create_protein_dataset(
  inputs: str | Path | Sequence[str | Path | IO[str]],
  batch_size: int,
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

  if hdf5_path is None:
    msg = "Preprocessing did not return a valid HDF5 path."
    raise ValueError(msg)

  source = sources.HDF5DataSource(hdf5_path)

  ds = grain.MapDataset.source(source)
  ds = ds.map(operations.LoadHDF5Frame(hdf5_path))

  ds = ds.to_iter_dataset()
  ds = ds.batch(batch_size, batch_fn=operations.pad_and_collate_proteins)

  performance_config = pick_performance_config(
    ds=ds,
    ram_budget_mb=1024,
    max_workers=None,
    max_buffer_size=None,
  )
  return ds.mp_prefetch(performance_config.multiprocessing_options)
