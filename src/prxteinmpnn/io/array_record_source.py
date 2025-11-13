"""Grain data source for loading preprocessed array_record files with physics features."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, SupportsIndex

import grain.python as grain
import msgpack
import msgpack_numpy as m
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader

from prxteinmpnn.utils.data_structures import ProteinTuple

m.patch()

logger = logging.getLogger(__name__)


class ArrayRecordDataSource(grain.RandomAccessDataSource):
  """Grain data source for preprocessed protein structures in array_record format.

  This source reads from array_record files created by the PQR preprocessing pipeline,
  which include precomputed physics features (electrostatic forces projected onto
  backbone frame).

  Attributes:
      array_record_path: Path to the .array_record file
      index: Dictionary mapping protein_id to record index
      reader: ArrayRecordReader instance
      _length: Total number of records

  Example:
      >>> source = ArrayRecordDataSource(
      ...     "data/preprocessed/train.array_record",
      ...     "data/preprocessed/train.index.json"
      ... )
      >>> protein = source[0]  # Returns ProteinTuple with physics_features
      >>> print(protein.physics_features.shape)  # (n_residues, 5)

  """

  def __init__(
    self,
    array_record_path: str | Path,
    index_path: str | Path,
  ) -> None:
    """Initialize the array_record data source.

    Args:
        array_record_path: Path to the array_record file
        index_path: Path to the JSON index file mapping protein_id -> index

    Raises:
        FileNotFoundError: If array_record or index file doesn't exist
        ValueError: If index file is malformed

    """
    super().__init__()
    self.array_record_path = Path(array_record_path)
    self.index_path = Path(index_path)

    if not self.array_record_path.exists():
      msg = f"Array record file not found: {self.array_record_path}"
      raise FileNotFoundError(msg)

    if not self.index_path.exists():
      msg = f"Index file not found: {self.index_path}"
      raise FileNotFoundError(msg)

    # Load index
    with self.index_path.open("r") as f:
      self.index = json.load(f)

    # Initialize reader
    self.reader = ArrayRecordReader(str(self.array_record_path))
    self._length = self.reader.num_records()

    # Validate index
    if len(self.index) != self._length:
      logger.warning(
        "Index size (%d) doesn't match record count (%d). Some proteins may be inaccessible.",
        len(self.index),
        self._length,
      )

    logger.info(
      "Loaded array_record data source with %d proteins from %s",
      self._length,
      self.array_record_path,
    )

  def __len__(self) -> int:
    """Return the total number of records."""
    return self._length

  def __getitem__(self, index: SupportsIndex) -> ProteinTuple:
    """Load and deserialize a protein structure.

    Args:
        index: Integer index of the record to load

    Returns:
        ProteinTuple with all fields including physics_features

    Raises:
        IndexError: If index is out of range
        RuntimeError: If deserialization fails

    """
    idx = int(index)
    if not 0 <= idx < len(self):
      msg = f"Index {idx} out of range [0, {len(self)})"
      raise IndexError(msg)

    try:
      # Read record
      record_bytes = self.reader.read(idx, idx + 1)[0]

      # Deserialize
      record_data = msgpack.unpackb(record_bytes, raw=False)

      # Convert to ProteinTuple
      return self._record_to_protein_tuple(record_data)

    except Exception as e:
      msg = f"Failed to load record at index {idx}"
      logger.exception(msg)
      raise RuntimeError(msg) from e

  def _record_to_protein_tuple(self, record: dict[str, Any]) -> ProteinTuple:
    """Convert deserialized record to ProteinTuple.

    Args:
        record: Dictionary with protein data

    Returns:
        ProteinTuple instance

    """
    # Extract all fields, converting to appropriate types
    return ProteinTuple(
      coordinates=np.array(record["coordinates"], dtype=np.float32),
      aatype=np.array(record["aatype"], dtype=np.int8),
      atom_mask=np.array(record["atom_mask"], dtype=bool),
      residue_index=np.array(record["residue_index"], dtype=np.int32),
      chain_index=np.array(record["chain_index"], dtype=np.int32),
      dihedrals=None,  # Not computed during preprocessing
      source=record["source_file"],
      # Full atomic data
      full_coordinates=np.array(record["full_coordinates"], dtype=np.float32),
      charges=np.array(record["charges"], dtype=np.float32),
      radii=np.array(record["radii"], dtype=np.float32),
      # Estat metadata
      estat_backbone_mask=np.array(record["estat_backbone_mask"], dtype=bool),
      estat_resid=np.array(record["estat_resid"], dtype=np.int32),
      estat_chain_index=np.array(record["estat_chain_index"], dtype=np.int32),
      # Physics features (the key addition!)
      physics_features=np.array(record["physics_features"], dtype=np.float32),
    )

  def close(self) -> None:
    """Close the ArrayRecordReader."""
    if hasattr(self, "reader"):
      self.reader.close()

  def __del__(self) -> None:
    """Cleanup when object is destroyed."""
    self.close()


def get_protein_by_id(
  source: ArrayRecordDataSource,
  protein_id: str,
) -> ProteinTuple | None:
  """Retrieve a protein by its ID.

  Args:
      source: ArrayRecordDataSource instance
      protein_id: Protein identifier

  Returns:
      ProteinTuple if found, None otherwise

  Example:
      >>> source = ArrayRecordDataSource("train.array_record", "train.index.json")
      >>> protein = get_protein_by_id(source, "1UBQ")

  """
  if protein_id not in source.index:
    logger.warning("Protein ID '%s' not found in index", protein_id)
    return None

  record_index = source.index[protein_id]
  return source[record_index]
