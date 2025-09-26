"""Data sources for loading protein structures using Grain.

This module defines the `RandomAccessDataSource` implementations that serve as the
starting point for a Grain data pipeline. It is responsible for identifying and
indexing various types of inputs, such as local files, PDB IDs, and FoldComp IDs.
"""

import pathlib
import re
import warnings
from collections.abc import Sequence
from typing import IO, Any, SupportsIndex

import grain.python as grain

from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

_FOLDCOMP_AFDB_PATTERN = re.compile(r"AF-[A-Z0-9]{6,10}-F[0-9]+-model_v[0-9]+")
_FOLDCOMP_ESM_PATTERN = re.compile(r"MGY[PC][0-9]{12,}(?:\.[0-9]+)?(?:_[0-9]+)?")
_FOLDCOMP_PATTERN = re.compile(
  rf"(?:{_FOLDCOMP_AFDB_PATTERN.pattern})|(?:{_FOLDCOMP_ESM_PATTERN.pattern})",
)
_PDB_PATTERN = re.compile(r"^[a-zA-Z0-9]{4}$")
_MD_CATH_PATTERN = re.compile(r"[a-zA-Z0-9]{5}[0-9]{2}")


class MixedInputDataSource(grain.RandomAccessDataSource):
  """A Grain DataSource that handles a heterogeneous list of protein structure inputs.

  This source can process a mix of local file paths, directories, PDB IDs,
  FoldComp IDs, and in-memory StringIO objects. It categorizes each input
  and provides a unified interface for downstream Grain operations.
  """

  def __init__(
    self,
    inputs: Sequence[str | IO[str]],
    foldcomp_database: FoldCompDatabase | None = None,
  ) -> None:
    """Initialize the data source by categorizing and indexing all inputs.

    Args:
        inputs: A sequence of input items.
        foldcomp_database: An optional FoldCompDatabase for resolving FoldComp IDs.

    """
    super().__init__()
    self.foldcomp_database = foldcomp_database
    self._items: list[tuple[str, Any]] = []
    self._categorize_inputs(inputs)

  def _categorize_inputs(self, inputs: Sequence[str | IO[str]]) -> None:
    """Iterate through raw inputs and categorize them for processing."""
    foldcomp_ids = []
    for item in inputs:
      if isinstance(item, str):
        self._categorize_string_input(item, foldcomp_ids)
      elif hasattr(item, "read"):  # StringIO-like
        self._items.append(("string_io", item))
      else:
        warnings.warn(f"Unsupported input type: {type(item)}", stacklevel=2)

    self._handle_foldcomp_ids(foldcomp_ids)

  def _categorize_string_input(self, item: str, foldcomp_ids: list[str]) -> None:
    """Categorize a string input as foldcomp ID, PDB ID, or file path."""
    if _FOLDCOMP_PATTERN.match(item):
      foldcomp_ids.append(item)
      return

    if _PDB_PATTERN.match(item) and not pathlib.Path(item).exists():
      self._items.append(("pdb_id", item))
      return

    if _MD_CATH_PATTERN.match(item) and not pathlib.Path(item).exists():
      self._items.append(("md_cath_id", item))
      return

    self._categorize_path_input(item)

  def _categorize_path_input(self, item: str) -> None:
    """Categorize a path input as directory or file."""
    path = pathlib.Path(item)
    if path.is_dir():
      self._items.extend(
        ("file_path", str(p))
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in {".pdb", ".cif"}
      )
    elif path.is_file():
      self._items.append(("file_path", str(path)))
    else:
      warnings.warn(f"Input '{item}' could not be categorized.", stacklevel=2)

  def _handle_foldcomp_ids(self, foldcomp_ids: list[str]) -> None:
    """Handle the collected FoldComp IDs."""
    if not foldcomp_ids:
      return

    if self.foldcomp_database:
      self._items.append(("foldcomp_ids", foldcomp_ids))
    else:
      warnings.warn(
        (
          "FoldComp IDs were provided but no database was configured. Configuring based on "
          "available IDs and matching to regex patterns."
        ),
        stacklevel=2,
      )
      if any(_FOLDCOMP_AFDB_PATTERN.match(fid) for fid in foldcomp_ids) and any(
        _FOLDCOMP_ESM_PATTERN.match(fid) for fid in foldcomp_ids
      ):
        self.foldcomp_database = "afesm_foldcomp"
        self._items.append(("foldcomp_ids", foldcomp_ids))
      elif any(_FOLDCOMP_AFDB_PATTERN.match(fid) for fid in foldcomp_ids):
        self.foldcomp_database = "afdb_uniprot_v4"
        self._items.append(("foldcomp_ids", foldcomp_ids))
      elif any(_FOLDCOMP_ESM_PATTERN.match(fid) for fid in foldcomp_ids):
        self.foldcomp_database = "esmatlas"
        self._items.append(("foldcomp_ids", foldcomp_ids))
      else:
        warnings.warn(
          "Could not determine appropriate FoldComp database for provided IDs.",
          stacklevel=2,
        )

  def __len__(self) -> int:
    """Return the total number of individual data records."""
    return len(self._items)

  def __getitem__(self, index: SupportsIndex) -> tuple[str, Any]:
    """Return the categorized input item at the given index.

    Args:
        index (SupportsIndex): The index of the item to retrieve.

    Returns:
        tuple[str, Any]: The categorized input item.

    Raises:
        IndexError: If the index is out of range.

    Example:
        >>> ds = MixedInputDataSource(["1abc.pdb"])
        >>> ds[0]
        ('file_path', '1abc.pdb')

    """
    idx = int(index)
    if not 0 <= idx < len(self):
      msg = f"Attempted to access index {idx}, but valid indices are 0 to {len(self) - 1}."
      raise IndexError(msg)
    return self._items[idx]
