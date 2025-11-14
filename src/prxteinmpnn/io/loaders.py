"""Provides a high-level API for creating Grain-based data loaders."""

import pathlib
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import IO, Any

import grain

from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

from . import dataset, operations, prefetch_autotune
from .array_record_source import ArrayRecordDataSource  # NEW


def create_protein_dataset(
  inputs: str | Path | Sequence[str | Path | IO[str]],
  batch_size: int,
  parse_kwargs: dict[str, Any] | None = None,
  foldcomp_database: FoldCompDatabase | None = None,
  pass_mode: str = "intra",  # noqa: S107
  *,
  use_electrostatics: bool = False,
  use_vdw: bool = False,
  use_preprocessed: bool = False,
  preprocessed_index_path: str | Path | None = None,
) -> grain.IterDataset:
  """Construct a high-performance protein data pipeline using Grain.

  Args:
      inputs: A single input (file, PDB ID, etc.) or a sequence of such inputs.
              When use_preprocessed=True, this should be the path to the array_record file.
      batch_size: The number of protein structures to include in each batch.
      parse_kwargs: Optional dictionary of keyword arguments for parsing.
      foldcomp_database: Optional path to a FoldComp database.
      pass_mode: "intra" (default) for normal batching, "inter" for concatenation.
      use_electrostatics: Whether to compute and include electrostatic features.
      use_vdw: Whether to compute and include van der Waals features.
      use_preprocessed: If True, load from preprocessed array_record files
      preprocessed_index_path: Path to index file (required if use_preprocessed=True)

  Returns:
      A Grain IterDataset that yields batches of padded `Protein` objects.

  Example:
      >>> # File-based loading (original)
      >>> ds = create_protein_dataset(
      ...     inputs="data/train/",
      ...     batch_size=8,
      ... )

      >>> # Preprocessed loading (new, faster)
      >>> ds = create_protein_dataset(
      ...     inputs="data/preprocessed/train.array_record",
      ...     batch_size=8,
      ...     use_preprocessed=True,
      ...     preprocessed_index_path="data/preprocessed/train.index.json",
      ... )

  """
  parse_kwargs = parse_kwargs or {}

  if use_preprocessed:
    if preprocessed_index_path is None:
      msg = "preprocessed_index_path is required when use_preprocessed=True"
      raise ValueError(msg)

    if not isinstance(inputs, (str, Path)):
      msg = "When use_preprocessed=True, inputs must be a single path to array_record file"
      raise ValueError(msg)

    source = ArrayRecordDataSource(
      array_record_path=inputs,
      index_path=preprocessed_index_path,
    )
    ds = grain.MapDataset.source(source)

  else:
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, pathlib.Path)):
      inputs = (inputs,)

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

  batch_fn = (
    operations.concatenate_proteins_for_inter_mode
    if pass_mode == "inter"  # noqa: S105
    else partial(
      operations.pad_and_collate_proteins,
      use_electrostatics=use_electrostatics,
      use_vdw=use_vdw,
    )
  )

  return ds.to_iter_dataset(read_options=performance_config.read_options).batch(
    batch_size,
    batch_fn=batch_fn,
  )
