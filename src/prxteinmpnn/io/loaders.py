"""Provides a high-level API for creating Grain-based data loaders."""

import pathlib
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

import grain

from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

from . import dataset, operations, prefetch_autotune


def create_protein_dataset(
  inputs: str | Path | Sequence[str | Path | IO[str]],
  batch_size: int,
  parse_kwargs: dict[str, Any] | None = None,
  foldcomp_database: FoldCompDatabase | None = None,
  pass_mode: str = "intra",  # noqa: S107
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
      pass_mode: "intra" (default) for normal batching, "inter" for concatenation.
      cache_path: Optional path to cache the preprocessed HDF5 file. If None,
                  a default path is used.

  Returns:
      A Grain IterDataset that yields batches of padded `Protein` objects.

  """
  if not isinstance(inputs, Sequence) or isinstance(inputs, (str, pathlib.Path)):
    inputs = (inputs,)

  parse_kwargs = parse_kwargs or {}

  source = dataset.ProteinDataSource(
    inputs=inputs,
    parse_kwargs=parse_kwargs,
    foldcomp_database=foldcomp_database,
  )
  ds = grain.MapDataset.source(source)

  performance_config = prefetch_autotune.pick_performance_config(
    ds=ds,
    ram_budget_mb=1024,
    max_workers=None,
    max_buffer_size=None,
  )

  # Choose batch function based on pass_mode
  batch_fn = (
    operations.concatenate_proteins_for_inter_mode
    if pass_mode == "inter"  # noqa: S105
    else operations.pad_and_collate_proteins
  )

  return ds.to_iter_dataset(read_options=performance_config.read_options).batch(
    batch_size,
    batch_fn=batch_fn,
  )
