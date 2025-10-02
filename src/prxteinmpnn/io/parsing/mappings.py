"""Utilities for mapping between different protein sequence and atom representations.

prxteinmpnn.io.parsing.mappings
"""

import logging
import pathlib
from collections.abc import Mapping

import numpy as np

from prxteinmpnn.utils.residue_constants import (
  atom_order,
  resname_to_idx,
  restype_order,
  restype_order_with_x,
  unk_restype_index,
)

logger = logging.getLogger(__name__)


MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
_AF_TO_MPNN_PERM = np.array(
  [MPNN_ALPHABET.index(k) for k in AF_ALPHABET],
)

_MPNN_TO_AF_PERM = np.array(
  [AF_ALPHABET.index(k) for k in MPNN_ALPHABET],
)


def af_to_mpnn(sequence: np.ndarray) -> np.ndarray:
  """Convert a sequence of integer indices from AlphaFold's to ProteinMPNN's alphabet order."""
  logger.debug("Converting sequence indices from AlphaFold to ProteinMPNN alphabet.")
  return _AF_TO_MPNN_PERM[sequence]


def mpnn_to_af(sequence: np.ndarray) -> np.ndarray:
  """Convert a sequence of integer indices from ProteinMPNN's to AlphaFold's alphabet order."""
  logger.debug("Converting sequence indices from ProteinMPNN to AlphaFold alphabet.")
  return _MPNN_TO_AF_PERM[sequence]


def _check_if_file_empty(file_path: str) -> bool:
  """Check if the file is empty."""
  logger.debug("Checking if file path %s is empty.", file_path)
  path = pathlib.Path(file_path)
  suffix = path.suffix.lower()
  try:
    with path.open() as f:
      if suffix in {".h5", ".hdf5"}:
        is_empty = not f.readable()
        if is_empty:
          logger.warning("HDF5 file path %s is not readable.", file_path)
        return is_empty

      # For text files
      is_empty = f.readable() and f.read().strip() == ""
      if is_empty:
        logger.warning("Text file path %s is readable but content is empty.", file_path)
      return is_empty
  except FileNotFoundError:
    logger.warning("File not found: %s", file_path)
    return True
  except Exception as e:
    logger.exception("Error checking if file %s is empty.", file_path, exc_info=e)
    return True


def string_key_to_index(
  string_keys: np.ndarray,
  key_map: Mapping[str, int],
  unk_index: int | None = None,
) -> np.ndarray:
  """Convert string keys to integer indices based on a mapping."""
  logger.debug("Converting %d string keys to integer indices.", len(string_keys))
  if unk_index is None:
    unk_index = len(key_map)

  sorted_keys = np.array(sorted(key_map.keys()))
  sorted_values = np.array([key_map[k] for k in sorted_keys])
  indices = np.searchsorted(sorted_keys, string_keys)
  indices = np.clip(indices, 0, len(sorted_keys) - 1)

  found_keys = sorted_keys[indices]
  is_known = found_keys == string_keys

  num_unknown = np.sum(~is_known)
  if num_unknown > 0:
    logger.debug("%d unknown keys encountered and mapped to index %d.", num_unknown, unk_index)

  return np.where(is_known, sorted_values[indices], unk_index)


def string_to_protein_sequence(
  sequence: str,
  aa_map: dict | None = None,
  unk_index: int | None = None,
) -> np.ndarray:
  """Convert a string sequence to a ProteinSequence."""
  logger.debug("Converting protein sequence string of length %d to indices.", len(sequence))
  if unk_index is None:
    unk_index = unk_restype_index

  if aa_map is None:
    aa_map = restype_order
    return af_to_mpnn(
      string_key_to_index(np.array(list(sequence), dtype="U3"), aa_map, unk_index),
    )
  return string_key_to_index(np.array(list(sequence), dtype="U3"), aa_map, unk_index)


def protein_sequence_to_string(
  sequence: np.ndarray,
  aa_map: dict | None = None,
) -> str:
  """Convert a ProteinSequence to a string."""
  logger.debug("Converting protein sequence indices of length %d to string.", len(sequence))
  if aa_map is None:
    aa_map = {i: aa for aa, i in restype_order_with_x.items()}

  af_seq = mpnn_to_af(sequence)

  return "".join([aa_map.get(int(aa), "X") for aa in af_seq])


def residue_names_to_aatype(
  residue_names: np.ndarray,
  aa_map: dict | None = None,
) -> np.ndarray:
  """Convert 3-letter residue names to amino acid type indices."""
  logger.debug("Converting %d 3-letter residue names to aatype indices.", len(residue_names))
  if aa_map is None:
    aa_map = resname_to_idx

  aa_indices = string_key_to_index(residue_names, aa_map, unk_restype_index)
  aa_indices = af_to_mpnn(aa_indices)
  return np.asarray(aa_indices, dtype=np.int8)


def atom_names_to_index(
  atom_names: np.ndarray,
  atom_map: dict | None = None,
) -> np.ndarray:
  """Convert atom names to atom type indices."""
  logger.debug("Converting %d atom names to atom type indices.", len(atom_names))
  if atom_map is None:
    atom_map = atom_order

  atom_indices = string_key_to_index(atom_names, atom_map, -1)
  return np.asarray(atom_indices)
