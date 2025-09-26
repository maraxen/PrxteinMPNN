"""Data operations for processing protein structures within a Grain pipeline.

This module implements `grain.transforms.Map` and `grain.IterOperation` classes
for parsing, transforming, and batching protein data.
"""

import pathlib
import warnings
from collections.abc import Sequence
from typing import cast

import grain
import h5py
import jax
import jax.numpy as jnp

from prxteinmpnn.utils.data_structures import Protein, ProteinTuple


class LoadHDF5Frame(grain.transforms.Map):
  """Load a single protein frame from a preprocessed HDF5 file by index.

  This operation is designed to work with `HDF5DataSource`. It receives an
  integer index and performs a direct slice from the HDF5 datasets to efficiently
  load the corresponding frame data.
  """

  def __init__(self, hdf5_path: str | pathlib.Path):
    """Initialize the operation with the path to the HDF5 file.

    Args:
        hdf5_path: Path to the preprocessed HDF5 file.

    """
    self.hdf5_path = hdf5_path
    self.h5_file: h5py.File | None = None
    self._dataset_keys: list[str] | None = None

  def _ensure_file_open(self) -> h5py.File:
    """Open the HDF5 file if it's not already open.

    This method is crucial for multiprocessing in Grain. Each worker process
    will get its own file handle, avoiding concurrency issues.
    """
    if self.h5_file is None:
      self.h5_file = h5py.File(self.hdf5_path, "r")
    return self.h5_file

  @property
  def dataset_keys(self) -> list[str]:
    """Cache the keys available in the HDF5 file."""
    if self._dataset_keys is None:
      f = self._ensure_file_open()
      self._dataset_keys = list(f.keys())
    return self._dataset_keys

  def map(self, index: int) -> ProteinTuple:  # type: ignore[override]
    """Read data for the given index from the HDF5 file and return a ProteinTuple.

    Args:
        index: The integer index of the frame to load.

    Returns:
        A ProteinTuple containing the data for the requested frame.

    """
    f = self._ensure_file_open()
    data = {}
    for key in self.dataset_keys:
      data[key] = cast("ProteinTuple", f[key])[index]

    # Decode byte strings back to Python strings
    if "source" in data and isinstance(data["source"], bytes):
      data["source"] = data["source"].decode("utf-8")

    # Ensure all fields from ProteinTuple are present, even if None
    all_fields = ProteinTuple.__annotations__.keys()
    for field in all_fields:
      if field not in data:
        data[field] = None

    return ProteinTuple(**data)


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
