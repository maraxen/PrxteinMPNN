"""Data operations for processing protein structures within a Grain pipeline.

This module implements `grain.MapOperation` and `grain.IterOperation` classes
for parsing, transforming, and batching protein data.
"""

import pathlib
import warnings
from collections.abc import Sequence
from io import StringIO
from typing import Any

import grain
import jax
import jax.numpy as jnp
import requests

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein, ProteinTuple
from prxteinmpnn.utils.foldcomp_utils import get_protein_structures


def _fetch_pdb(pdb_id: str) -> str:
  """Fetch PDB content from the RCSB data bank.

  Args:
    pdb_id (str): The PDB identifier.

  Returns:
    str: The PDB file content as a string.

  Raises:
    requests.HTTPError: If the HTTP request fails.
    requests.RequestException: For other request-related errors.

  Example:
    >>> content = _fetch_pdb("1ABC")

  """
  url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
  response = requests.get(url, timeout=60)
  response.raise_for_status()
  return response.text


def _fetch_md_cath(md_cath_id: str) -> pathlib.Path:
  """Fetch h5 content from the MD-CATH data bank and save to disk."""
  url = f"https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data/mdcath_dataset_{md_cath_id}.h5"
  response = requests.get(url, timeout=60)
  response.raise_for_status()
  data_dir = pathlib.Path("mdcath_data")
  data_dir.mkdir(exist_ok=True)
  md_cath_file = data_dir / f"mdcath_dataset_{md_cath_id}.h5"
  with md_cath_file.open("wb") as f:
    f.write(response.content)
  return md_cath_file


class ParseStructure(grain.transforms.Map):
  """Parse a protein structure from various sources.

  This Grain MapOperation takes a categorized input from `MixedInputDataSource`
  and dispatches it to the correct parsing function.

  Args:
    parse_kwargs (Optional[dict[str, Any]]): Additional keyword arguments for parsing.

  Example:
    >>> op = ParseStructure()
    >>> result = op.map(("file_path", "example.pdb"))

  """

  def __init__(self, parse_kwargs: dict[str, Any] | None = None) -> None:
    """Initialize the ParseStructure operation."""
    super().__init__()
    self.parse_kwargs: dict[str, Any] = parse_kwargs or {}

  def map(self, element: tuple[str, Any]) -> list[ProteinTuple] | None:  # type: ignore[override]  # noqa: PLR0911
    """Parse a single categorized input element.

    Args:
      element (tuple[str, Any]): The input type and value.

    Returns:
      Optional[list[ProteinTuple]]: List of ProteinTuples, or None if parsing fails.

    Raises:
      None: Warnings are issued instead of raising exceptions.

    Example:
      >>> op = ParseStructure()
      >>> op.map(("file_path", "example.pdb"))

    """
    input_type, value = element
    try:
      if input_type == "file_path":
        return list(parse_input(value, **self.parse_kwargs))
      if input_type == "pdb_id":
        pdb_content = _fetch_pdb(value)
        return list(parse_input(StringIO(pdb_content), **self.parse_kwargs))
      if input_type == "string_io":
        return list(parse_input(value, **self.parse_kwargs))
      if input_type == "md_cath_id":
        md_cath_file = _fetch_md_cath(value)
        return list(parse_input(md_cath_file, **self.parse_kwargs))
      if input_type == "foldcomp_ids":
        return list(get_protein_structures(value))
    except Exception as e:  # noqa: BLE001
      warnings.warn(f"Failed to parse {input_type} '{value}': {e}", stacklevel=2)
      return None
    return None


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
