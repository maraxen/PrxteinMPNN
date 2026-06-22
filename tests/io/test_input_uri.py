"""Tests for URI parsing (input_uri module)."""

from __future__ import annotations

import pytest
import typer

from aminx.io.input_uri import ParsedInput, get_dedup_key, parse_input_uri


class TestParseInputUriBarePaths:
  """Test bare local paths (no scheme)."""

  def test_bare_absolute_path(self) -> None:
    """Bare absolute path -> local scheme."""
    result = parse_input_uri("/data/1ubq.pdb")
    assert result == ParsedInput(scheme="local", accession="/data/1ubq.pdb", fmt=None)

  def test_bare_relative_path(self) -> None:
    """Bare relative path -> local scheme."""
    result = parse_input_uri("./mydir")
    assert result == ParsedInput(scheme="local", accession="./mydir", fmt=None)

  def test_bare_glob(self) -> None:
    """Bare glob pattern -> local scheme."""
    result = parse_input_uri("*.cif")
    assert result == ParsedInput(scheme="local", accession="*.cif", fmt=None)

  def test_windows_path_not_scheme(self) -> None:
    """Windows path C:\\ is NOT a scheme (: not followed by //)."""
    result = parse_input_uri("C:\\x\\y.pdb")
    assert result == ParsedInput(scheme="local", accession="C:\\x\\y.pdb", fmt=None)


class TestParseInputUriFileScheme:
  """Test file:// scheme."""

  def test_file_absolute_path(self) -> None:
    """file:///path -> file scheme, /path accession."""
    result = parse_input_uri("file:///data/1ubq.pdb")
    assert result == ParsedInput(scheme="file", accession="/data/1ubq.pdb", fmt=None)

  def test_file_with_host_stripped(self) -> None:
    """file://host/path -> strips host, returns /path."""
    result = parse_input_uri("file://localhost/data/x.pdb")
    assert result == ParsedInput(scheme="file", accession="/data/x.pdb", fmt=None)

  def test_file_with_arbitrary_host_stripped(self) -> None:
    """file://myhost/some/path -> strips myhost."""
    result = parse_input_uri("file://myhost/some/path")
    assert result == ParsedInput(scheme="file", accession="/some/path", fmt=None)

  def test_file_empty_accession(self) -> None:
    """file:// with no accession -> error."""
    with pytest.raises(typer.BadParameter, match="has no accession"):
      parse_input_uri("file://")

  def test_file_host_no_path(self) -> None:
    """file://host with no path after host -> error."""
    with pytest.raises(typer.BadParameter, match="requires a path"):
      parse_input_uri("file://localhost")


class TestParseInputUriPDBScheme:
  """Test pdb:// scheme."""

  def test_pdb_default_format(self) -> None:
    """pdb://1A3A -> mmcif format (default)."""
    result = parse_input_uri("pdb://1A3A")
    assert result == ParsedInput(scheme="pdb", accession="1A3A", fmt="mmcif")

  def test_pdb_with_pdb_suffix(self) -> None:
    """pdb://1A3A.pdb -> pdb format."""
    result = parse_input_uri("pdb://1A3A.pdb")
    assert result == ParsedInput(scheme="pdb", accession="1A3A", fmt="pdb")

  def test_pdb_with_cif_suffix(self) -> None:
    """pdb://1A3A.cif -> mmcif format."""
    result = parse_input_uri("pdb://1A3A.cif")
    assert result == ParsedInput(scheme="pdb", accession="1A3A", fmt="mmcif")

  def test_pdb_empty_accession(self) -> None:
    """pdb:// with no accession -> error."""
    with pytest.raises(typer.BadParameter, match="has no accession"):
      parse_input_uri("pdb://")

  def test_pdb_unknown_format_suffix(self) -> None:
    """pdb://1A3A.unknown -> error."""
    with pytest.raises(typer.BadParameter, match="unknown format suffix"):
      parse_input_uri("pdb://1A3A.unknown")

  def test_pdb_empty_after_suffix(self) -> None:
    """pdb://.pdb -> error (empty accession after removing suffix)."""
    with pytest.raises(typer.BadParameter, match="has no accession"):
      parse_input_uri("pdb://.pdb")


class TestParseInputUriAFDBScheme:
  """Test afdb:// scheme."""

  def test_afdb_uniprot_id(self) -> None:
    """afdb://P12345 -> afdb scheme, P12345 accession."""
    result = parse_input_uri("afdb://P12345")
    assert result == ParsedInput(scheme="afdb", accession="P12345", fmt=None)

  def test_afdb_empty_accession(self) -> None:
    """afdb:// with no accession -> error."""
    with pytest.raises(typer.BadParameter, match="has no accession"):
      parse_input_uri("afdb://")


class TestParseInputUriMDCATHScheme:
  """Test mdcath:// scheme."""

  def test_mdcath_id(self) -> None:
    """mdcath://1abcA00 -> mdcath scheme, 1abcA00 accession."""
    result = parse_input_uri("mdcath://1abcA00")
    assert result == ParsedInput(scheme="mdcath", accession="1abcA00", fmt=None)

  def test_mdcath_empty_accession(self) -> None:
    """mdcath:// with no accession -> error."""
    with pytest.raises(typer.BadParameter, match="has no accession"):
      parse_input_uri("mdcath://")


class TestParseInputUriUnsupportedSchemes:
  """Test that unsupported schemes raise errors (no silent fallback)."""

  def test_s3_scheme_error(self) -> None:
    """s3://bucket/key -> error, no silent local fallback."""
    with pytest.raises(typer.BadParameter, match="no fetcher for scheme 's3://'"):
      parse_input_uri("s3://bucket/key")

  def test_http_scheme_error(self) -> None:
    """http://example.com -> error."""
    with pytest.raises(typer.BadParameter, match="no fetcher for scheme 'http://'"):
      parse_input_uri("http://example.com")

  def test_https_scheme_error(self) -> None:
    """https://example.com -> error."""
    with pytest.raises(typer.BadParameter, match="no fetcher for scheme 'https://'"):
      parse_input_uri("https://example.com")

  def test_gs_scheme_error(self) -> None:
    """gs://bucket/key -> error."""
    with pytest.raises(typer.BadParameter, match="no fetcher for scheme 'gs://'"):
      parse_input_uri("gs://bucket/key")


class TestGetDedupKey:
  """Test dedup key generation."""

  def test_dedup_key_local(self) -> None:
    """Dedup key for local path."""
    parsed = ParsedInput(scheme="local", accession="/data/x.pdb", fmt=None)
    key = get_dedup_key(parsed)
    assert key == ("local", "/data/x.pdb", None)

  def test_dedup_key_pdb_mmcif(self) -> None:
    """Dedup key for pdb://1A3A -> (pdb, 1A3A, mmcif)."""
    parsed = ParsedInput(scheme="pdb", accession="1A3A", fmt="mmcif")
    key = get_dedup_key(parsed)
    assert key == ("pdb", "1A3A", "mmcif")

  def test_dedup_key_pdb_pdb_format(self) -> None:
    """Dedup key for pdb://1A3A.pdb -> (pdb, 1A3A, pdb)."""
    parsed = ParsedInput(scheme="pdb", accession="1A3A", fmt="pdb")
    key = get_dedup_key(parsed)
    assert key == ("pdb", "1A3A", "pdb")

  def test_dedup_key_afdb(self) -> None:
    """Dedup key for afdb://P12345."""
    parsed = ParsedInput(scheme="afdb", accession="P12345", fmt=None)
    key = get_dedup_key(parsed)
    assert key == ("afdb", "P12345", None)

  def test_dedup_key_mdcath(self) -> None:
    """Dedup key for mdcath://1abcA00."""
    parsed = ParsedInput(scheme="mdcath", accession="1abcA00", fmt=None)
    key = get_dedup_key(parsed)
    assert key == ("mdcath", "1abcA00", None)


class TestParseInputUriEdgeCases:
  """Test edge cases and boundary conditions."""

  def test_uppercase_scheme_not_recognized(self) -> None:
    """Uppercase scheme like PDB:// is NOT recognized (scheme is lowercased in regex)."""
    # PDB:// does not match ^[a-z]... so it's treated as a bare path
    result = parse_input_uri("PDB://1A3A")
    assert result == ParsedInput(scheme="local", accession="PDB://1A3A", fmt=None)

  def test_mixed_case_bare_path(self) -> None:
    """Mixed case bare path treated as local."""
    result = parse_input_uri("MyPath.pdb")
    assert result == ParsedInput(scheme="local", accession="MyPath.pdb", fmt=None)

  def test_file_relative_path(self) -> None:
    """file://relative/path -> treated as file://host/path, strips 'relative'."""
    result = parse_input_uri("file://relative/path")
    # This is parsed as file://host/path where host='relative'
    # The spec strips the host, leaving '/path'
    assert result == ParsedInput(scheme="file", accession="/path", fmt=None)

  def test_scheme_with_plus_numbers_dash(self) -> None:
    """Valid scheme characters: a-z, 0-9, +, ., - (RFC 3986)."""
    # Test that a valid scheme with these chars is recognized (would need a fetcher)
    with pytest.raises(typer.BadParameter, match="no fetcher"):
      parse_input_uri("my-scheme+2.0://something")
