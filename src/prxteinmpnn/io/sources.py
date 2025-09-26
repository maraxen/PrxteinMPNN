"""Data sources for loading protein structures using Grain.

This module defines the `RandomAccessDataSource` implementations that serve as the
starting point for a Grain data pipeline. It is responsible for indexing various
types of inputs, such as local files, PDB IDs, and FoldComp IDs.
"""

import pathlib
from typing import SupportsIndex

import grain.python as grain
import h5py


class HDF5DataSource(grain.RandomAccessDataSource):
  """A Grain DataSource for reading preprocessed HDF5 files.

  This source opens a HDF5 file created by `preprocess_inputs_to_hdf5`, determines the
  total number of frames, and provides simple integer indices to downstream operations.
  """

  def __init__(
    self,
    hdf5_path: str | pathlib.Path,
  ) -> None:
    """Initialize the data source by reading the number of frames from the HDF5 file.

    Args:
        hdf5_path: Path to the preprocessed HDF5 file.

    """
    super().__init__()
    self.hdf5_path = hdf5_path
    with h5py.File(self.hdf5_path, "r") as f:
      if "coordinates" in f:
        self._length = f["coordinates"].shape[0]  # type: ignore[attr-access]
      else:
        # Handle empty or malformed files
        self._length = 0

  def __len__(self) -> int:
    """Return the total number of frames in the HDF5 file."""
    return self._length

  def __getitem__(self, index: SupportsIndex) -> int:
    """Return the integer index.

    The actual data loading is handled by a downstream MapOperation. This source
    only provides the indices.

    Args:
        index (SupportsIndex): The index of the item to retrieve.

    Returns:
        int: The index itself.

    Raises:
        IndexError: If the index is out of range.

    """
    idx = int(index)
    if not 0 <= idx < len(self):
      msg = f"Attempted to access index {idx}, but valid indices are 0 to {len(self) - 1}."
      raise IndexError(msg)
    return idx
