"""Fetch wrappers for remote sources (RCSB, AlphaFold, MD-CATH) with aminx-controlled cache.

The cache layout is:
  cache_dir/<scheme>/<accession>[/<fmt>]/

Cache HIT = subdir exists and contains >= 1 non-empty file; no fetcher call.
Cache MISS = subdir absent or empty; fetcher called, subdir populated.

Partial-download cleanup (M1): on any fetcher exception, the per-accession subdir
is deleted before re-raising as InputResolutionError.

Categorized error messages (T7): exceptions are classified by type/message pattern
(404/not-found vs network/offline vs other) to provide distinct, actionable hints.

Mock boundary (M4): fetchers are called via module attribute (proxide.io.fetch_*),
not imported directly, so tests can patch aminx.io.proxide_fetch.proxide.io.*.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import proxide.io

if TYPE_CHECKING:
  from .input_uri import PDBFormat

logger = logging.getLogger(__name__)


class InputResolutionError(Exception):
  """Typed error for input-resolution failures (fetch, cache, network, offline).

  Raised when:
  - A fetcher fails (network, offline, 404, malformed)
  - Cache mkdir/write fails (permission, full-disk)
  - Post-resolution validation fails

  Callers (resolver, runner guard) catch this and emit actionable messages.
  """


def _categorize_fetch_error(exc: Exception, uri: str) -> str:
  """Categorize a fetcher exception and return an actionable error message.

  Args:
    exc: The exception raised by the fetcher.
    uri: The URI being fetched (e.g., 'pdb://1A3A' or 'afdb://P12345').

  Returns:
    A categorized, actionable error message.
  """
  exc_type = type(exc).__name__
  exc_msg = str(exc).lower()

  # Classify by exception type and message pattern
  is_network_error = isinstance(
    exc,
    (
      ConnectionError,
      TimeoutError,
      OSError,
    ),
  ) or any(
    kw in exc_msg
    for kw in [
      "connection",
      "timeout",
      "network",
      "unreachable",
      "refused",
      "reset",
    ]
  )

  is_not_found_error = any(
    kw in exc_msg
    for kw in [
      "404",
      "not found",
      "does not exist",
      "no structure",
    ]
  )

  # Return categorized message
  if is_not_found_error:
    return (
      f"no structure found for {uri} — the accession may be invalid or "
      f"not available in the remote database. verify the accession and "
      f"check the source documentation (RCSB/AlphaFold/MD-CATH)."
    )
  if is_network_error:
    return (
      f"network error fetching {uri}: {exc_type}. remote fetches require "
      f"network access at CLI/submit time. resolve on a connected host, "
      f"then submit with the cached local path (cluster compute nodes are offline)."
    )
  return (
    f"unexpected error fetching {uri}: {exc_type}. remote fetches require "
    f"network access at CLI/submit time. resolve on a connected host, "
    f"then submit with the cached local path."
  )


def fetch_pdb(
  pdb_id: str,
  cache_dir: Path | str,
  fmt: PDBFormat = "mmcif",
) -> Path:
  """Fetch a PDB structure from RCSB with aminx-controlled cache.

  Args:
    pdb_id: 4-letter PDB accession (e.g., '1A3A').
    cache_dir: Root cache directory.
    fmt: 'pdb' or 'mmcif' (default).

  Returns:
    Path to the cached/fetched file.

  Raises:
    InputResolutionError: On fetch failure, I/O error, or offline.
  """
  cache_dir = Path(cache_dir)
  subdir = cache_dir / "pdb" / pdb_id / fmt

  # Check for cache hit
  hit_path = _check_cache_hit(subdir)
  if hit_path:
    return hit_path

  # Cache miss: create subdir, fetch, cache result
  try:
    subdir.mkdir(parents=True, exist_ok=True)
  except (PermissionError, OSError) as exc:
    raise InputResolutionError(
      f"cannot write input cache at {cache_dir}: {exc.__class__.__name__} — "
      f"check permissions / free space",
    ) from exc

  try:
    # M4: module attribute call for test patchability
    fetched_path = proxide.io.fetch_rcsb(pdb_id, output_dir=str(subdir), format_type=fmt)
    return Path(fetched_path)
  except Exception as exc:
    # M1: partial-download cleanup before re-raise
    _cleanup_subdir(subdir)
    # T7: categorized error message by failure type
    uri = f"pdb://{pdb_id}"
    message = _categorize_fetch_error(exc, uri)
    raise InputResolutionError(message) from exc


def fetch_afdb(
  uniprot_id: str,
  cache_dir: Path | str,
) -> Path:
  """Fetch an AlphaFold structure from AfDB with aminx-controlled cache.

  Args:
    uniprot_id: UniProt accession (e.g., 'P12345').
    cache_dir: Root cache directory.

  Returns:
    Path to the cached/fetched file.

  Raises:
    InputResolutionError: On fetch failure, I/O error, or offline.
  """
  cache_dir = Path(cache_dir)
  subdir = cache_dir / "afdb" / uniprot_id

  # Check for cache hit
  hit_path = _check_cache_hit(subdir)
  if hit_path:
    return hit_path

  # Cache miss: create subdir, fetch, cache result
  try:
    subdir.mkdir(parents=True, exist_ok=True)
  except (PermissionError, OSError) as exc:
    raise InputResolutionError(
      f"cannot write input cache at {cache_dir}: {exc.__class__.__name__} — "
      f"check permissions / free space",
    ) from exc

  try:
    # M4: module attribute call for test patchability
    fetched_path = proxide.io.fetch_afdb(uniprot_id, output_dir=str(subdir), version=4)
    return Path(fetched_path)
  except Exception as exc:
    # M1: partial-download cleanup before re-raise
    _cleanup_subdir(subdir)
    # T7: categorized error message by failure type
    uri = f"afdb://{uniprot_id}"
    message = _categorize_fetch_error(exc, uri)
    raise InputResolutionError(message) from exc


def fetch_md_cath(
  md_cath_id: str,
  cache_dir: Path | str,
) -> Path:
  """Fetch an MD-CATH structure with aminx-controlled cache.

  Args:
    md_cath_id: MD-CATH accession (e.g., '1abcA00').
    cache_dir: Root cache directory.

  Returns:
    Path to the cached/fetched .h5 file.

  Raises:
    InputResolutionError: On fetch failure, I/O error, or offline.
  """
  cache_dir = Path(cache_dir)
  subdir = cache_dir / "mdcath" / md_cath_id

  # Check for cache hit
  hit_path = _check_cache_hit(subdir)
  if hit_path:
    return hit_path

  # Cache miss: create subdir, fetch, cache result
  try:
    subdir.mkdir(parents=True, exist_ok=True)
  except (PermissionError, OSError) as exc:
    raise InputResolutionError(
      f"cannot write input cache at {cache_dir}: {exc.__class__.__name__} — "
      f"check permissions / free space",
    ) from exc

  try:
    # M4: module attribute call for test patchability
    fetched_path = proxide.io.fetch_md_cath(md_cath_id, output_dir=str(subdir))
    return Path(fetched_path)
  except Exception as exc:
    # M1: partial-download cleanup before re-raise
    _cleanup_subdir(subdir)
    # T7: categorized error message by failure type
    uri = f"mdcath://{md_cath_id}"
    message = _categorize_fetch_error(exc, uri)
    raise InputResolutionError(message) from exc


def _check_cache_hit(subdir: Path) -> Path | None:
  """Check if cache subdir exists and contains >= 1 non-empty file (B1 cache hit).

  Args:
    subdir: Per-accession cache subdirectory.

  Returns:
    Path to the (first) cached file, or None if miss.
  """
  if not subdir.exists():
    return None

  # Check for >= 1 non-empty file in subdir
  for item in subdir.iterdir():
    if item.is_file() and item.stat().st_size > 0:
      return item

  return None


def _cleanup_subdir(subdir: Path) -> None:
  """Delete a per-accession subdir (used on partial-download failure, M1).

  Args:
    subdir: Per-accession cache subdirectory to delete.
  """
  try:
    if subdir.exists():
      shutil.rmtree(subdir)
  except OSError as exc:
    # Best-effort cleanup; log but don't raise on cleanup failure
    logger.warning("cleanup failed for %s: %s", subdir, exc)
