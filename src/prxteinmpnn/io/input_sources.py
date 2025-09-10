"""Handlers for various input sources to load protein structures."""

import pathlib
import re
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from typing import Any

import aiohttp
import anyio

from prxteinmpnn.utils.data_structures import Protein, ProteinEnsemble
from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase, get_protein_structures

from .parsing import parse_input


class InputSource(ABC):
  """Abstract base class for an input source handler."""

  def __init__(self, value: Any, **kwargs: dict[str, Any]) -> None:  # noqa: ANN401
    """Initialize with the input value and any additional parameters."""
    self.value = value
    self.kwargs = kwargs

  @abstractmethod
  async def process(self, executor: ProcessPoolExecutor) -> ProteinEnsemble:
    """Process the input source and yield Protein."""
    raise NotImplementedError
    yield  # This makes this method an async generator


class FilePathSource(InputSource):
  """Handles processing of a single structure file."""

  async def process(self, executor: ProcessPoolExecutor) -> ProteinEnsemble:
    """Process a single structure file."""
    try:

      def _parse() -> list[Protein]:
        future = executor.submit(parse_input, self.value, **self.kwargs)  # type: ignore[arg-type]
        return future.result()

      proteins = await anyio.to_thread.run_sync(_parse)  # type: ignore[attr-access]
      for protein in proteins:
        yield protein, self.value
    except Exception as e:  # noqa: BLE001
      warnings.warn(f"Failed to process file '{self.value}': {e}", stacklevel=2)


class DirectorySource(InputSource):
  """Handles recursive processing of directories."""

  async def process(self, executor: ProcessPoolExecutor) -> ProteinEnsemble:
    """Recursively process all structure files in a directory."""

    async def _find_files() -> list[pathlib.Path]:
      """Asynchronously find all files in the directory."""
      return await anyio.to_thread.run_sync(lambda: list(self.value.rglob("*")))  # type: ignore[attr-access]

    files = await _find_files()
    for file_path in files:
      if file_path.is_file() and file_path.suffix in [".pdb", ".cif", ".xtc", ".dcd"]:
        handler = FilePathSource(str(file_path), **self.kwargs)
        async for protein in handler.process(executor):
          yield protein


class PDBIDSource(InputSource):
  """Handles fetching and processing of PDB IDs from RCSB."""

  async def process(self, executor: ProcessPoolExecutor) -> ProteinEnsemble:  # noqa: ARG002
    """Fetch and process a PDB structure by its ID."""
    url = f"https://files.rcsb.org/download/{self.value}.pdb"
    try:
      async with aiohttp.ClientSession() as session, session.get(url) as response:
        response.raise_for_status()
        content = await response.text()

        proteins = await anyio.to_thread.run_sync(parse_input, StringIO(content), **self.kwargs)  # type: ignore[attr-access]
        for protein in proteins:
          yield protein, self.value
    except Exception as e:  # noqa: BLE001
      warnings.warn(f"Failed to fetch or process PDB ID '{self.value}': {e}", stacklevel=2)


class StringIOSource(InputSource):
  """Handles processing of in-memory StringIO objects."""

  async def process(self, executor: ProcessPoolExecutor) -> ProteinEnsemble:  # noqa: ARG002
    """Process a StringIO object containing structure data."""
    try:
      proteins = await anyio.to_thread.run_sync(parse_input, self.value, **self.kwargs)  # type: ignore[attr-access]
      for protein in proteins:
        yield protein, "StringIO"
    except Exception as e:  # noqa: BLE001
      warnings.warn(f"Failed to process StringIO input: {e}", stacklevel=2)


class FoldCompSource(InputSource):
  """Handles fetching structures from a FoldComp database."""

  def __init__(
    self,
    value: list[str],
    foldcomp_database: FoldCompDatabase,
    **kwargs: dict[str, Any],
  ) -> None:
    """Initialize with a list of FoldComp IDs and a database reference."""
    super().__init__(value, **kwargs)
    self.db: FoldCompDatabase = foldcomp_database

  async def process(self, executor: ProcessPoolExecutor) -> ProteinEnsemble:
    """Fetch and process structures from FoldComp by their IDs."""
    if not self.db:
      warnings.warn("FoldComp IDs provided but no database specified.", stacklevel=2)
      return
    try:

      def _get_structures() -> list[Protein]:
        future = executor.submit(get_protein_structures, self.value, self.db)
        return future.result()

      proteins = await anyio.to_thread.run_sync(_get_structures)  # type: ignore[attr-access]
      for protein in proteins:
        yield protein, self.value
    except Exception as e:  # noqa: BLE001
      warnings.warn(f"Failed to process FoldComp IDs: {e}", stacklevel=2)


_FOLDCOMP_PATTERN = re.compile(r"AF-[A-Z0-9]{6,10}-[A-Z0-9]{1,2}-model_v[0-9]+")
_PDB_PATTERN = re.compile(r"^[a-zA-Z0-9]{4}$")


def get_source_handler(item: str | StringIO, **kwargs: dict[str, Any]) -> InputSource | None:
  """Determine the correct source handler for an input."""
  if isinstance(item, StringIO):
    return StringIOSource(item, **kwargs)
  if isinstance(item, str):
    if _FOLDCOMP_PATTERN.match(item):
      return None
    if _PDB_PATTERN.match(item) and not pathlib.Path(item).exists():
      return PDBIDSource(item, **kwargs)

    path = pathlib.Path(item)
    if path.is_dir():
      return DirectorySource(path, **kwargs)
    if path.is_file():
      return FilePathSource(str(path), **kwargs)

  warnings.warn(f"Input '{item}' could not be categorized and will be ignored.", stacklevel=2)
  return None
