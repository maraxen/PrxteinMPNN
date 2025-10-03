"""Data loading utilities for protein structure data."""

import logging
import pathlib
from collections.abc import Iterator, Sequence
from typing import IO, Any

import grain.python as grain

from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

from .process import frame_iterator_from_inputs

# Instantiate the logger
logger = logging.getLogger(__name__)


class FrameDataset(grain.IterDataset):
  """A Grain DataSource for streaming protein frames from various input sources.

  This source implements an iterator that yields frames one by one, avoiding
  loading the entire dataset into memory.
  """

  def __init__(
    self,
    inputs: Sequence[str | pathlib.Path | IO[str]],
    parse_kwargs: dict[str, Any] | None = None,
    foldcomp_database: FoldCompDatabase | None = None,
  ) -> None:
    """Initialize the data source with arguments needed to create the iterator.

    Args:
        inputs: A sequence of mixed input types (file paths, PDB IDs, etc.).
        parse_kwargs: Optional dictionary of keyword arguments for parsing.
        foldcomp_database: An optional FoldCompDatabase for resolving FoldComp IDs.

    """
    super().__init__()
    self.inputs = inputs
    self.parse_kwargs = parse_kwargs
    self.foldcomp_database = foldcomp_database

  def __iter__(self) -> Iterator[ProteinTuple]:  # type: ignore[override]
    """Create and yield from the frame iterator.

    This method is called by Grain to start streaming data.
    """
    with frame_iterator_from_inputs(
      self.inputs,
      self.parse_kwargs,
      self.foldcomp_database,  # type: ignore[arg-type]
    ) as frame_iterator:
      yield from frame_iterator
