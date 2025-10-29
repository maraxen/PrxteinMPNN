"""Pre-process various input formats into a single HDF5 file for efficient loading."""

import logging
import pathlib
import re
import time
import warnings
from collections.abc import Generator, Iterator, Sequence
from io import StringIO
from typing import IO, Any

import requests

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase, get_protein_structures

# Instantiate the logger
logger = logging.getLogger(__name__)

# --- Regex patterns for ID matching ---
_FOLDCOMP_AFDB_PATTERN = re.compile(r"AF-[A-Z0-9]{6,10}-F[0-9]+-model_v[0-9]+")
_FOLDCOMP_ESM_PATTERN = re.compile(r"MGY[PC][0-9]{12,}(?:\.[0-9]+)?(?:_[0-9]+)?")
_FOLDCOMP_PATTERN = re.compile(
  rf"(?:{_FOLDCOMP_AFDB_PATTERN.pattern})|(?:{_FOLDCOMP_ESM_PATTERN.pattern})",
)
_PDB_PATTERN = re.compile(r"^[a-zA-Z0-9]{4}$")
_MD_CATH_PATTERN = re.compile(r"[a-zA-Z0-9]{5}[0-9]{2}")


def _fetch_with_retry(
  url: str,
  max_retries: int = 3,
  initial_delay: float = 1.0,
  backoff_factor: float = 2.0,
  timeout: int = 60,
) -> requests.Response:
  """Fetch content from a URL with exponential backoff retry logic.

  Args:
    url: The URL to fetch.
    max_retries: Maximum number of retry attempts (default: 3).
    initial_delay: Initial delay in seconds before first retry (default: 1.0).
    backoff_factor: Multiplier for delay between retries (default: 2.0).
    timeout: Request timeout in seconds (default: 60).

  Returns:
    The response object if successful.

  Raises:
    requests.RequestException: If all retry attempts fail.

  Example:
    >>> response = _fetch_with_retry("https://example.com/data.pdb")
    >>> content = response.text

  """
  delay = initial_delay
  last_exception = None

  for attempt in range(max_retries):
    try:
      response = requests.get(url, timeout=timeout)
      response.raise_for_status()
    except requests.RequestException as e:
      last_exception = e
      if attempt < max_retries - 1:
        logger.warning(
          "Attempt %d/%d failed for URL %s: %s. Retrying in %.1f seconds...",
          attempt + 1,
          max_retries,
          url,
          e,
          delay,
        )
        time.sleep(delay)
        delay *= backoff_factor
      else:
        logger.exception("All %d attempts failed for URL %s", max_retries, url)
    else:
      return response

  msg = f"Failed to fetch {url} after {max_retries} attempts"
  raise requests.RequestException(msg) from last_exception


def _fetch_pdb(pdb_id: str) -> str:
  """Fetch PDB content from the RCSB data bank with retry logic.

  Args:
    pdb_id: The PDB identifier (e.g., "1abc").

  Returns:
    The PDB file content as a string.

  Raises:
    requests.RequestException: If fetching fails after all retry attempts.

  Example:
    >>> pdb_content = _fetch_pdb("1abc")

  """
  url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
  response = _fetch_with_retry(url)
  return response.text


def _fetch_md_cath(md_cath_id: str) -> pathlib.Path:
  """Fetch h5 content from the MD-CATH data bank and save to disk with retry logic.

  Args:
    md_cath_id: The MD-CATH identifier (e.g., "1a2b00").

  Returns:
    Path to the downloaded HDF5 file.

  Raises:
    requests.RequestException: If fetching fails after all retry attempts.

  Example:
    >>> path = _fetch_md_cath("1a2b00")

  """
  url = f"https://huggingface.co/datasets/compsciencelab/mdCATH/resolve/main/data/mdcath_dataset_{md_cath_id}.h5"
  response = _fetch_with_retry(url)
  data_dir = pathlib.Path("mdcath_data")
  data_dir.mkdir(exist_ok=True)
  md_cath_file = data_dir / f"mdcath_dataset_{md_cath_id}.h5"
  with md_cath_file.open("wb") as f:
    f.write(response.content)
  return md_cath_file


def _resolve_inputs(  # noqa: C901
  inputs: Sequence[str | IO[str] | pathlib.Path],
  foldcomp_database: FoldCompDatabase | None = None,
) -> Generator[str | pathlib.Path | IO[str] | ProteinTuple, None, None]:
  """Resolve a heterogeneous list of inputs into parseable sources.

  This generator categorizes each input and yields a source that `parse_input`
  can directly handle (file paths or StringIO objects). It fetches data for
  PDB, MD-CATH, and FoldComp IDs.

  Args:
      inputs: A sequence of input items.
      foldcomp_database: An optional FoldCompDatabase for resolving FoldComp IDs.

  Yields:
      A parseable source (str, pathlib.Path, or StringIO).

  """
  foldcomp_ids = []
  for item in inputs:
    try:
      if isinstance(item, str):
        if _FOLDCOMP_PATTERN.match(item):
          foldcomp_ids.append(item)
          continue
        if _PDB_PATTERN.match(item) and not pathlib.Path(item).exists():
          yield StringIO(_fetch_pdb(item))
          continue
        if _MD_CATH_PATTERN.match(item) and not pathlib.Path(item).exists():
          yield _fetch_md_cath(item)
          continue

        path = pathlib.Path(item)
        if path.is_file():
          yield path
        elif path.is_dir():
          yield from (p for p in path.rglob("*") if p.is_file())
        else:
          warnings.warn(f"Input string '{item}' could not be categorized.", stacklevel=2)
      elif hasattr(item, "read"):  # StringIO-like
        yield item
      else:
        warnings.warn(f"Unsupported input type: {type(item)}", stacklevel=2)
    except Exception as e:  # noqa: BLE001
      warnings.warn(f"Failed to resolve input '{item}': {e}", stacklevel=2)

  if foldcomp_ids:
    yield from get_protein_structures(foldcomp_ids, foldcomp_database)


def frame_iterator_from_inputs(
  inputs: Sequence[str | pathlib.Path | IO[str]],
  parse_kwargs: dict[str, Any] | None = None,
  foldcomp_database: FoldCompDatabase | None = None,
) -> Iterator[ProteinTuple]:
  """Create a generator that yields ProteinTuple frames from mixed inputs.

  This function resolves various input types (file paths, PDB IDs, FoldComp IDs,
  StringIO objects, and direct ProteinTuple instances) and parses them into
  ProteinTuple frames.

  Args:
    inputs: A sequence of mixed input types.
    parse_kwargs: Optional dictionary of keyword arguments for the parsing function.
    foldcomp_database: An optional FoldCompDatabase for resolving FoldComp IDs.

  Yields:
    ProteinTuple frames parsed from the inputs.

  """
  parse_kwargs = parse_kwargs or {}
  resolved_sources = _resolve_inputs(inputs, foldcomp_database)
  for source in resolved_sources:
    if isinstance(source, ProteinTuple):
      yield source
    else:
      yield from parse_input(source, **parse_kwargs)
