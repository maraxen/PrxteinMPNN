"""Data loading utilities for protein structure data."""

import logging
import pathlib
from collections.abc import Sequence
from typing import IO, Any, SupportsIndex

import grain.python as grain

from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.foldcomp_utils import (
  FoldCompDatabase,
)

from .process import frame_iterator_from_inputs

logger = logging.getLogger(__name__)


class ProteinDataSource(grain.RandomAccessDataSource):
  """Implements a Grain DataSource for streaming protein structure frames."""

  def __init__(
    self,
    inputs: Sequence[str | pathlib.Path | IO[str]],
    parse_kwargs: dict[str, Any] | None = None,
    foldcomp_database: FoldCompDatabase | None = None,
  ) -> None:
    """Initialize the data source by preparing the frame iterator.

    Args:
        inputs: A sequence of input sources (file paths, file-like objects, etc.).
        parse_kwargs: Optional keyword arguments to pass to the parsing function.
        foldcomp_database: An optional FoldCompDatabase for resolving FoldComp IDs.

    """
    super().__init__()
    self.inputs = inputs
    self.parse_kwargs = parse_kwargs or {}
    self.foldcomp_database = foldcomp_database
    self.frames = list(
      frame_iterator_from_inputs(
        self.inputs,
        self.parse_kwargs,
        self.foldcomp_database,
      ),
    )
    self._length = len(self.frames)

  def __len__(self) -> int:
    """Return the total number of frames available."""
    return self._length

  def __getitem__(self, index: SupportsIndex) -> ProteinTuple:
    """Return the ProteinTuple at the specified index.

    Args:
        index (SupportsIndex): The index of the item to retrieve.

    Returns:
        ProteinTuple: The protein structure frame at the specified index.

    Raises:
        IndexError: If the index is out of range.

    """
    idx = int(index)
    if not 0 <= idx < len(self):
      msg = f"Attempted to access index {idx}, but valid indices are 0 to {len(self) - 1}."
      raise IndexError(msg)
    return self.frames[idx]
