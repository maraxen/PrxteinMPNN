"""Handlers for various input sources to load protein structures."""

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from typing import Any

import anyio

from prxteinmpnn.utils.data_structures import ProteinEnsemble
from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

from .input_sources import (
  _FOLDCOMP_PATTERN,
  FoldCompSource,
  InputSource,
  get_source_handler,
)


async def load(
  inputs: Sequence[str | StringIO] | str | StringIO,
  foldcomp_database: FoldCompDatabase | None = None,
  max_concurrency: int = 10,
  **kwargs: dict[str, Any],
) -> ProteinEnsemble:
  """Asynchronously loads model inputs from various sources with high efficiency.

  This function processes inputs concurrently, using a process pool for CPU-bound
  parsing tasks and non-blocking I/O for network and file access.

  Args:
      inputs: A single or sequence of inputs. Can be file paths, directory paths,
              PDB IDs, FoldComp IDs, or StringIO objects.
      foldcomp_database: A FoldCompDatabase instance for resolving FoldComp IDs.
      max_concurrency: The maximum number of inputs to process concurrently.
      **kwargs: Additional keyword arguments passed to the parsing function.

  """
  if isinstance(inputs, (str, StringIO)):
    inputs = [inputs]

  foldcomp_ids = [
    item for item in inputs if isinstance(item, str) and _FOLDCOMP_PATTERN.match(item)
  ]
  other_inputs = [
    item for item in inputs if not (isinstance(item, str) and _FOLDCOMP_PATTERN.match(item))
  ]

  handlers = [get_source_handler(item, **kwargs) for item in other_inputs]
  if foldcomp_ids and foldcomp_database:
    handlers.append(FoldCompSource(foldcomp_ids, foldcomp_database, **kwargs))

  valid_handlers = [h for h in handlers if h is not None]

  with ProcessPoolExecutor() as executor:
    semaphore = anyio.Semaphore(max_concurrency)

    async def process_with_semaphore(
      handler: InputSource,
    ) -> ProteinEnsemble:
      async with semaphore:
        async for protein in handler.process(executor):
          yield protein

    for handler in valid_handlers:
      async for protein in process_with_semaphore(handler):
        yield protein
