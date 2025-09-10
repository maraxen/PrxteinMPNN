"""Handlers for various input sources to load protein structures."""

from collections.abc import Sequence
from io import StringIO
from typing import Any

import anyio

from prxteinmpnn.utils.data_structures import ProteinTuple
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
) -> tuple[list[ProteinTuple], list[str]]:
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

  items = []
  limiter = anyio.CapacityLimiter(max_concurrency)

  async def process_and_collect(handler: InputSource, results: list) -> None:
    """Process an input source and collect results."""
    async with limiter:
      results.extend([protein async for protein in handler.process()])

  async with anyio.create_task_group() as tg:
    for handler in valid_handlers:
      tg.start_soon(process_and_collect, handler, items)

  if not items:
    msg = "Cannot batch an empty ProteinEnsemble."
    raise ValueError(msg)

  _proteins, sources = zip(*items, strict=False)

  if not isinstance(sources, list):
    sources = list(sources)

  if not isinstance(_proteins, list):
    _proteins = list(_proteins)

  return _proteins, sources
