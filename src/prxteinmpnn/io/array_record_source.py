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
from array_record.python.array_record_module import (  # type: ignore[unresolved-import]
  ArrayRecordReader,
)

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
    split: str = "train",
  ) -> None:
    """Initialize the array_record data source.

    Args:
        array_record_path: Path to the array_record file
        index_path: Path to the JSON index file mapping protein_id -> index
        split: Data split to load ("train", "valid", "test")

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

    # Initialize reader early to get num_records if needed for "all" or "inference" split
    self.reader = ArrayRecordReader(str(self.array_record_path))
    self._length = self.reader.num_records()

    # Load index
    # If index_path doesn't exist, we might be in a special "inference" or "all" mode
    # where the index is implicitly all records in the array_record file.
    if not self.index_path.exists():
      if split in ["inference", "all"]:
        logger.info(
            "Index file not found at %s, but split is '%s'. "
            "Assuming all records in array_record file for this split.",
            self.index_path,
            split,
        )
        # Create a dummy full_index where all records belong to this split
        full_index = {
            f"record_{i}": {"idx": [i], "set": split}
            for i in range(self._length)
        }
      else:
        msg = f"Index file not found: {self.index_path}"
        raise FileNotFoundError(msg)
    else:
      with self.index_path.open("r") as f:
        full_index = json.load(f)

    # Filter index by split
    self.index = {}
    for pid, entry in full_index.items():
      if entry.get("set") == split:
        # The new format is {"idx": [i1, i2...], "set": "..."}
        # We flatten the list of indices for this split
        for _ in entry["idx"]:
          # We use a composite key if needed, but here we just need a list of record indices
          # For RandomAccessDataSource, we need a mapping from 0..N to record_index
          pass
        self.index[pid] = entry

    # Create a linear mapping from 0..N to file record indices
    self._record_indices = []
    for entry in self.index.values():
      self._record_indices.extend(entry["idx"])

    # Fallback for inference/all split if no records found in index
    if len(self._record_indices) == 0 and split in ["inference", "all"]:
      logger.info(
        "No records found for split '%s' in index, but split is '%s'. "
        "Falling back to using all records in array_record file.",
        split,
        split,
      )
      self._record_indices = list(range(self._length))

    logger.info(
      "Loaded %d records for split '%s' from %s",
      len(self._record_indices),
      split,
      self.index_path,
    )

    # Initialize reader
    self.reader = ArrayRecordReader(str(self.array_record_path))
    self._length = self.reader.num_records()

    # Validate index
    if len(self._record_indices) == 0:
      logger.warning("No records found for split '%s'", split)

    logger.info(
      "Loaded array_record data source with %d proteins from %s",
      self._length,
      self.array_record_path,
    )

  def __len__(self) -> int:
    """Return the total number of records."""
    return len(self._record_indices)

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
      record_idx = self._record_indices[idx]
      record_bytes = self.reader.read(record_idx, record_idx + 1)[0]

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
    # Extract all fields, converting to appropriate types
    # Use .get() for optional fields to ensure robustness

    # Basic structure
    coordinates = np.array(record["coordinates"], dtype=np.float32)
    n_residues = coordinates.shape[0]

    # Defaults for physics features if missing
    default_physics = np.zeros((n_residues, 5), dtype=np.float32)

    # Defaults for full atomic data if missing
    # Note: We don't know n_atoms easily without full_coordinates,
    # but if full_coordinates is missing, we probably don't need charges/radii either.
    # We'll return empty arrays if full_coordinates is missing.
    full_coords = record.get("full_coordinates")
    if full_coords is not None:
        full_coords = np.array(full_coords, dtype=np.float32)
        n_atoms = full_coords.shape[0]
        default_charges = np.zeros((n_atoms,), dtype=np.float32)
        default_radii = np.zeros((n_atoms,), dtype=np.float32)
    else:
        full_coords = np.zeros((0, 3), dtype=np.float32)
        default_charges = np.zeros((0,), dtype=np.float32)
        default_radii = np.zeros((0,), dtype=np.float32)

    return ProteinTuple(
      coordinates=coordinates,
      aatype=np.array(record["aatype"], dtype=np.int8),
      atom_mask=np.array(
          record.get("atom_mask", np.ones((n_residues, 37), dtype=bool)),
          dtype=bool,
      ),
      residue_index=np.array(
          record.get("residue_index", np.arange(n_residues)),
          dtype=np.int32,
      ),
      chain_index=np.array(
          record.get("chain_index", np.zeros(n_residues)),
          dtype=np.int32,
      ),
      dihedrals=None,
      source=record.get("source_file", "unknown"),

      # Full atomic data
      full_coordinates=full_coords,
      charges=np.array(record.get("charges", default_charges), dtype=np.float32),
      radii=np.array(record.get("radii", default_radii), dtype=np.float32),

      # Estat metadata
      estat_backbone_mask=np.array(
          record.get(
              "estat_backbone_mask",
              record.get("atom_mask", np.zeros(n_residues, dtype=bool)),
          ),
          dtype=bool,
      ),
      estat_resid=np.array(
          record.get("estat_resid", record.get("residue_index", np.zeros(n_residues))),
          dtype=np.int32,
      ),
      estat_chain_index=np.array(
          record.get("estat_chain_index", record.get("chain_index", np.zeros(n_residues))),
          dtype=np.int32,
      ),

      # Physics features
      physics_features=np.array(record.get("physics_features", default_physics), dtype=np.float32),
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

  if protein_id not in source.index:
    logger.warning("Protein ID '%s' not found in index for this split", protein_id)
    return None

  # Get the first record index for this protein
  # Note: This assumes we want the first conformation if multiple exist
  file_record_index = source.index[protein_id]["idx"][0]

  # We need to find where this file_record_index is in our filtered _record_indices list
  # This is O(N) which is slow, but get_protein_by_id is mainly for debugging
  try:
      dataset_index = source._record_indices.index(file_record_index)  # noqa: SLF001
      return source[dataset_index]
  except ValueError:
      logger.warning(
          "Protein ID '%s' found in index but its records are not in this split",
          protein_id,
      )
      return None
