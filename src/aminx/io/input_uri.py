"""Pure URI parser for heterogeneous input tokens (local, pdb://, afdb://, mdcath://)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import typer

InputScheme = Literal["file", "pdb", "afdb", "mdcath", "local"]
PDBFormat = Literal["pdb", "mmcif"]


@dataclass(frozen=True)
class ParsedInput:
  """Result of parsing an input token into scheme, accession, and format.

  Attributes:
    scheme: One of 'local', 'file', 'pdb', 'afdb', 'mdcath'.
    accession: The identifier/path component (empty for bare local paths).
    fmt: Format hint ('pdb' or 'mmcif' for pdb scheme, None for others).
  """

  scheme: InputScheme
  accession: str
  fmt: PDBFormat | None = None


def parse_input_uri(token: str) -> ParsedInput:
  """Parse an input token into scheme, accession, and format.

  Handles:
    - Bare paths: `/data/1ubq.pdb`, `./mydir`, `*.cif` -> (local, path, None)
    - file:// scheme: `file:///data/1ubq.pdb` -> (file, /data/1ubq.pdb, None)
    - file://host/path: strips host, -> (file, /path, None)
    - pdb://X: -> (pdb, X, mmcif)
    - pdb://X.pdb: -> (pdb, X, pdb)
    - pdb://X.cif: -> (pdb, X, mmcif)
    - afdb://X: -> (afdb, X, None)
    - mdcath://X: -> (mdcath, X, None)

  Raises:
    typer.BadParameter: For empty accession, unknown scheme, invalid format suffix,
      or unsupported schemes (s3, http, https, etc.).

  Args:
    token: Input string to parse.

  Returns:
    ParsedInput with scheme, accession, and format.
  """
  # Check for URI scheme: ^[a-z][a-z0-9+.-]*://
  scheme_match = re.match(r"^([a-z][a-z0-9+.-]*)://", token)

  if not scheme_match:
    # No scheme -> bare local path
    return ParsedInput(scheme="local", accession=token, fmt=None)

  scheme = scheme_match.group(1)
  remainder = token[len(scheme) + 3 :]  # Skip "scheme://"

  # Handle each scheme
  if scheme == "file":
    return _parse_file_scheme(remainder)
  if scheme == "pdb":
    return _parse_pdb_scheme(remainder)
  if scheme == "afdb":
    return _parse_afdb_scheme(remainder)
  if scheme == "mdcath":
    return _parse_mdcath_scheme(remainder)
  # Known grammar but no fetcher -> error, no silent fallback
  raise typer.BadParameter(f"no fetcher for scheme '{scheme}://'")


def _parse_file_scheme(remainder: str) -> ParsedInput:
  """Parse file:// scheme.

  Handles file:///path and file://host/path (strips host).
  """
  if not remainder:
    raise typer.BadParameter("scheme 'file://' has no accession")

  # RFC 8089: file:///path (local) or file://host/path (host-based)
  # After removing "scheme://", we have:
  # - file:///path → remainder = "/path" (starts with /)
  # - file://host/path → remainder = "host/path" (no leading /)

  if remainder.startswith("/"):
    # file:///path (local file, empty authority)
    path = remainder
  else:
    # file://host/path (host-based file); strip the host
    first_slash = remainder.find("/")
    if first_slash == -1:
      # file://host with no path
      raise typer.BadParameter("file://host format requires a path after host")
    # Keep only the path part (from first / onward)
    path = remainder[first_slash:]

  if not path:
    raise typer.BadParameter("scheme 'file://' has no accession")

  return ParsedInput(scheme="file", accession=path, fmt=None)


def _parse_pdb_scheme(remainder: str) -> ParsedInput:
  """Parse pdb:// scheme with optional format suffix (.pdb or .cif)."""
  if not remainder:
    raise typer.BadParameter("scheme 'pdb://' has no accession")

  # Check for format suffix
  fmt: PDBFormat = "mmcif"  # default
  accession = remainder

  if remainder.endswith(".pdb"):
    fmt = "pdb"
    accession = remainder[:-4]  # Remove .pdb
  elif remainder.endswith(".cif"):
    fmt = "mmcif"
    accession = remainder[:-4]  # Remove .cif
  elif "." in remainder and remainder.rsplit(".", maxsplit=1)[-1] not in ["pdb", "cif"]:
    # Unknown format suffix
    ext = remainder.rsplit(".", maxsplit=1)[-1]
    raise typer.BadParameter(f"unknown format suffix '.{ext}' for pdb:// (use .pdb or .cif)")

  if not accession:
    raise typer.BadParameter("scheme 'pdb://' has no accession")

  return ParsedInput(scheme="pdb", accession=accession, fmt=fmt)


def _parse_afdb_scheme(remainder: str) -> ParsedInput:
  """Parse afdb:// scheme."""
  if not remainder:
    raise typer.BadParameter("scheme 'afdb://' has no accession")

  return ParsedInput(scheme="afdb", accession=remainder, fmt=None)


def _parse_mdcath_scheme(remainder: str) -> ParsedInput:
  """Parse mdcath:// scheme."""
  if not remainder:
    raise typer.BadParameter("scheme 'mdcath://' has no accession")

  return ParsedInput(scheme="mdcath", accession=remainder, fmt=None)


def get_dedup_key(parsed: ParsedInput) -> tuple[str, str, str | None]:
  """Return the dedup key for pre-resolution deduplication.

  For (scheme, accession, fmt) uniqueness. Local paths dedup by normalized path.

  Args:
    parsed: ParsedInput result.

  Returns:
    Tuple of (scheme, accession, fmt) for use as dict key.
  """
  return (parsed.scheme, parsed.accession, parsed.fmt)
