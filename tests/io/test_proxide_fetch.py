"""Tests for proxide fetch wrappers and aminx-controlled cache (T2, gates G3/G4)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aminx.io.proxide_fetch import (
    InputResolutionError,
    fetch_afdb,
    fetch_md_cath,
    fetch_pdb,
)


class TestFetchPdb:
  """Tests for fetch_pdb (PDB via RCSB)."""

  def test_fetch_pdb_cache_miss_calls_fetcher_once(self, tmp_path: Path) -> None:
    """G3: Cache MISS calls fetcher once."""
    cache_dir = tmp_path / "cache"
    accession = "1A3A"
    fmt = "mmcif"

    # Mock the fetcher to create artifact on miss
    def mock_fetch(pdb_id: str, output_dir: str, format_type: str) -> str:
      subdir = Path(output_dir)
      artifact = subdir / "1A3A.cif"
      artifact.write_text("mock pdb data")
      return str(artifact)

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_rcsb", side_effect=mock_fetch) as mock:
      result = fetch_pdb(accession, cache_dir, fmt)
      assert result.name == "1A3A.cif"
      assert mock.call_count == 1

  def test_fetch_pdb_cache_hit_skips_fetcher(self, tmp_path: Path) -> None:
    """G3: Cache HIT returns cached path and calls fetcher ZERO times."""
    cache_dir = tmp_path / "cache"
    accession = "1A3A"
    fmt = "mmcif"

    # Populate cache with existing artifact
    subdir = cache_dir / "pdb" / accession / fmt
    subdir.mkdir(parents=True, exist_ok=True)
    artifact = subdir / "1A3A.cif"
    artifact.write_text("cached pdb data")

    # Mock fetcher (should never be called)
    with patch("aminx.io.proxide_fetch.proxide.io.fetch_rcsb") as mock:
      result = fetch_pdb(accession, cache_dir, fmt)
      assert result == artifact
      assert mock.call_count == 0  # G3: zero-fetch on hit

  def test_fetch_pdb_cache_hit_with_empty_file_is_miss(self, tmp_path: Path) -> None:
    """Cache HIT requires ≥1 non-empty file (empty file = MISS)."""
    cache_dir = tmp_path / "cache"
    accession = "1A3A"
    fmt = "mmcif"

    # Populate cache with EMPTY artifact (should be treated as MISS)
    subdir = cache_dir / "pdb" / accession / fmt
    subdir.mkdir(parents=True, exist_ok=True)
    artifact = subdir / "1A3A.cif"
    artifact.write_text("")  # Empty!

    def mock_fetch(pdb_id: str, output_dir: str, format_type: str) -> str:
      # Simulate fetcher writing the real artifact
      real_artifact = subdir / "1A3A.cif"
      real_artifact.write_text("real data")
      return str(real_artifact)

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_rcsb", side_effect=mock_fetch) as mock:
      result = fetch_pdb(accession, cache_dir, fmt)
      # Should have called fetcher because empty file is not a hit
      assert mock.call_count == 1
      assert result == artifact

  def test_fetch_pdb_cache_dir_creation(self, tmp_path: Path) -> None:
    """Cache subdir is created on MISS."""
    cache_dir = tmp_path / "cache"
    accession = "1A3A"
    fmt = "pdb"

    def mock_fetch(pdb_id: str, output_dir: str, format_type: str) -> str:
      # Simulate fetcher writing artifact
      artifact = Path(output_dir) / "1A3A.pdb"
      artifact.write_text("pdb data")
      return str(artifact)

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_rcsb", side_effect=mock_fetch):
      result = fetch_pdb(accession, cache_dir, fmt)
      subdir = cache_dir / "pdb" / accession / fmt
      assert subdir.exists()
      assert result.parent == subdir

  def test_fetch_pdb_mkdir_permission_error(self, tmp_path: Path) -> None:
    """m2: mkdir PermissionError -> InputResolutionError."""
    cache_dir = tmp_path / "cache"
    accession = "1A3A"
    fmt = "mmcif"

    # Mock mkdir to raise PermissionError
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
      with pytest.raises(InputResolutionError) as exc_info:
        fetch_pdb(accession, cache_dir, fmt)
      assert "cannot write input cache" in str(exc_info.value)
      assert "PermissionError" in str(exc_info.value)

  def test_fetch_pdb_fetcher_raises_cleanup_subdir(self, tmp_path: Path) -> None:
    """M1: Fetcher exception -> cleanup partial artifact -> InputResolutionError."""
    cache_dir = tmp_path / "cache"
    accession = "1A3A"
    fmt = "mmcif"

    def mock_fetch_raises(pdb_id: str, output_dir: str, format_type: str) -> str:
      # Simulate writing a partial artifact then failing
      partial = Path(output_dir) / "1A3A.cif"
      partial.write_text("partial data")
      raise RuntimeError("Network error")

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_rcsb", side_effect=mock_fetch_raises):
      with pytest.raises(InputResolutionError) as exc_info:
        fetch_pdb(accession, cache_dir, fmt)

      assert "failed to fetch pdb://1A3A" in str(exc_info.value)
      assert "offline" in str(exc_info.value).lower()

      # M1: subdir should be deleted (cleanup)
      subdir = cache_dir / "pdb" / accession / fmt
      assert not subdir.exists(), "Partial artifact subdir should be cleaned up"


class TestFetchAfdb:
  """Tests for fetch_afdb (AlphaFold)."""

  def test_fetch_afdb_cache_miss_calls_fetcher_once(self, tmp_path: Path) -> None:
    """G3: Cache MISS calls fetcher once."""
    cache_dir = tmp_path / "cache"
    uniprot_id = "P12345"

    def mock_fetch(uniprot_id: str, output_dir: str, version: int) -> str:
      subdir = Path(output_dir)
      artifact = subdir / "P12345.pdb"
      artifact.write_text("afdb data")
      return str(artifact)

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_afdb", side_effect=mock_fetch) as mock:
      result = fetch_afdb(uniprot_id, cache_dir)
      assert result.name == "P12345.pdb"
      assert mock.call_count == 1

  def test_fetch_afdb_cache_hit_skips_fetcher(self, tmp_path: Path) -> None:
    """G3: Cache HIT returns cached path and calls fetcher ZERO times."""
    cache_dir = tmp_path / "cache"
    uniprot_id = "P12345"

    subdir = cache_dir / "afdb" / uniprot_id
    subdir.mkdir(parents=True, exist_ok=True)
    artifact = subdir / "P12345.pdb"
    artifact.write_text("cached afdb data")

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_afdb") as mock:
      result = fetch_afdb(uniprot_id, cache_dir)
      assert result == artifact
      assert mock.call_count == 0  # G3: zero-fetch on hit

  def test_fetch_afdb_fetcher_raises_cleanup_subdir(self, tmp_path: Path) -> None:
    """M1: Fetcher exception -> cleanup partial artifact -> InputResolutionError."""
    cache_dir = tmp_path / "cache"
    uniprot_id = "P12345"

    def mock_fetch_raises(uniprot_id: str, output_dir: str, version: int) -> str:
      partial = Path(output_dir) / "P12345.pdb"
      partial.write_text("partial")
      raise RuntimeError("404 not found")

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_afdb", side_effect=mock_fetch_raises):
      with pytest.raises(InputResolutionError) as exc_info:
        fetch_afdb(uniprot_id, cache_dir)

      assert "failed to fetch afdb://P12345" in str(exc_info.value)

      subdir = cache_dir / "afdb" / uniprot_id
      assert not subdir.exists(), "Partial artifact subdir should be cleaned up"

  def test_fetch_afdb_cache_hit_with_empty_file_is_miss(self, tmp_path: Path) -> None:
    """Cache HIT requires ≥1 non-empty file (empty file = MISS) [T6 breadth]."""
    cache_dir = tmp_path / "cache"
    uniprot_id = "P12345"

    # Populate cache with EMPTY artifact (should be treated as MISS)
    subdir = cache_dir / "afdb" / uniprot_id
    subdir.mkdir(parents=True, exist_ok=True)
    artifact = subdir / "P12345.pdb"
    artifact.write_text("")  # Empty!

    def mock_fetch(uniprot_id: str, output_dir: str, version: int) -> str:
      # Simulate fetcher writing the real artifact
      real_artifact = subdir / "P12345.pdb"
      real_artifact.write_text("real afdb data")
      return str(real_artifact)

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_afdb", side_effect=mock_fetch) as mock:
      result = fetch_afdb(uniprot_id, cache_dir)
      # Should have called fetcher because empty file is not a hit
      assert mock.call_count == 1
      assert result == artifact

  def test_fetch_afdb_cache_dir_creation(self, tmp_path: Path) -> None:
    """Cache subdir is created on MISS [T6 breadth]."""
    cache_dir = tmp_path / "cache"
    uniprot_id = "P12345"

    def mock_fetch(uniprot_id: str, output_dir: str, version: int) -> str:
      # Simulate fetcher writing artifact
      artifact = Path(output_dir) / "P12345.pdb"
      artifact.write_text("afdb data")
      return str(artifact)

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_afdb", side_effect=mock_fetch):
      result = fetch_afdb(uniprot_id, cache_dir)
      subdir = cache_dir / "afdb" / uniprot_id
      assert subdir.exists()
      assert result.parent == subdir

  def test_fetch_afdb_mkdir_permission_error(self, tmp_path: Path) -> None:
    """m2: mkdir PermissionError -> InputResolutionError [T6 breadth]."""
    cache_dir = tmp_path / "cache"
    uniprot_id = "P12345"

    # Mock mkdir to raise PermissionError
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
      with pytest.raises(InputResolutionError) as exc_info:
        fetch_afdb(uniprot_id, cache_dir)
      assert "cannot write input cache" in str(exc_info.value)
      assert "PermissionError" in str(exc_info.value)


class TestFetchMdCath:
  """Tests for fetch_md_cath (MD-CATH)."""

  def test_fetch_md_cath_cache_miss_calls_fetcher_once(self, tmp_path: Path) -> None:
    """G3: Cache MISS calls fetcher once."""
    cache_dir = tmp_path / "cache"
    md_cath_id = "1abcA00"

    def mock_fetch(md_cath_id: str, output_dir: str) -> str:
      subdir = Path(output_dir)
      artifact = subdir / "1abcA00.h5"
      artifact.write_text("h5 data")
      return str(artifact)

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_md_cath", side_effect=mock_fetch) as mock:
      result = fetch_md_cath(md_cath_id, cache_dir)
      assert result.name == "1abcA00.h5"
      assert mock.call_count == 1

  def test_fetch_md_cath_cache_hit_skips_fetcher(self, tmp_path: Path) -> None:
    """G3: Cache HIT returns cached path and calls fetcher ZERO times."""
    cache_dir = tmp_path / "cache"
    md_cath_id = "1abcA00"

    subdir = cache_dir / "mdcath" / md_cath_id
    subdir.mkdir(parents=True, exist_ok=True)
    artifact = subdir / "1abcA00.h5"
    artifact.write_text("cached h5 data")

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_md_cath") as mock:
      result = fetch_md_cath(md_cath_id, cache_dir)
      assert result == artifact
      assert mock.call_count == 0  # G3: zero-fetch on hit

  def test_fetch_md_cath_fetcher_raises_cleanup_subdir(self, tmp_path: Path) -> None:
    """M1: Fetcher exception -> cleanup partial artifact -> InputResolutionError."""
    cache_dir = tmp_path / "cache"
    md_cath_id = "1abcA00"

    def mock_fetch_raises(md_cath_id: str, output_dir: str) -> str:
      partial = Path(output_dir) / "1abcA00.h5"
      partial.write_text("partial h5")
      raise RuntimeError("Download failed")

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_md_cath", side_effect=mock_fetch_raises):
      with pytest.raises(InputResolutionError) as exc_info:
        fetch_md_cath(md_cath_id, cache_dir)

      assert "failed to fetch mdcath://1abcA00" in str(exc_info.value)

      subdir = cache_dir / "mdcath" / md_cath_id
      assert not subdir.exists(), "Partial artifact subdir should be cleaned up"

  def test_fetch_md_cath_cache_hit_with_empty_file_is_miss(self, tmp_path: Path) -> None:
    """Cache HIT requires ≥1 non-empty file (empty file = MISS) [T6 breadth]."""
    cache_dir = tmp_path / "cache"
    md_cath_id = "1abcA00"

    # Populate cache with EMPTY artifact (should be treated as MISS)
    subdir = cache_dir / "mdcath" / md_cath_id
    subdir.mkdir(parents=True, exist_ok=True)
    artifact = subdir / "1abcA00.h5"
    artifact.write_text("")  # Empty!

    def mock_fetch(md_cath_id: str, output_dir: str) -> str:
      # Simulate fetcher writing the real artifact
      real_artifact = subdir / "1abcA00.h5"
      real_artifact.write_text("real h5 data")
      return str(real_artifact)

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_md_cath", side_effect=mock_fetch) as mock:
      result = fetch_md_cath(md_cath_id, cache_dir)
      # Should have called fetcher because empty file is not a hit
      assert mock.call_count == 1
      assert result == artifact

  def test_fetch_md_cath_cache_dir_creation(self, tmp_path: Path) -> None:
    """Cache subdir is created on MISS [T6 breadth]."""
    cache_dir = tmp_path / "cache"
    md_cath_id = "1abcA00"

    def mock_fetch(md_cath_id: str, output_dir: str) -> str:
      # Simulate fetcher writing artifact
      artifact = Path(output_dir) / "1abcA00.h5"
      artifact.write_text("mdcath data")
      return str(artifact)

    with patch("aminx.io.proxide_fetch.proxide.io.fetch_md_cath", side_effect=mock_fetch):
      result = fetch_md_cath(md_cath_id, cache_dir)
      subdir = cache_dir / "mdcath" / md_cath_id
      assert subdir.exists()
      assert result.parent == subdir

  def test_fetch_md_cath_mkdir_permission_error(self, tmp_path: Path) -> None:
    """m2: mkdir PermissionError -> InputResolutionError [T6 breadth]."""
    cache_dir = tmp_path / "cache"
    md_cath_id = "1abcA00"

    # Mock mkdir to raise PermissionError
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
      with pytest.raises(InputResolutionError) as exc_info:
        fetch_md_cath(md_cath_id, cache_dir)
      assert "cannot write input cache" in str(exc_info.value)
      assert "PermissionError" in str(exc_info.value)


class TestUnwritableCacheDir:
  """Tests for m2: mkdir/write error mapping."""

  def test_full_disk_oslog_error(self, tmp_path: Path) -> None:
    """m2: OSError (full disk) -> InputResolutionError."""
    cache_dir = tmp_path / "cache"
    accession = "1A3A"

    with patch("aminx.io.proxide_fetch.Path.mkdir", side_effect=OSError("No space left")):
      with pytest.raises(InputResolutionError) as exc_info:
        fetch_pdb(accession, cache_dir)
      assert "cannot write input cache" in str(exc_info.value)
      assert "OSError" in str(exc_info.value)

  def test_full_disk_afdb_error(self, tmp_path: Path) -> None:
    """m2: OSError (full disk) on afdb -> InputResolutionError [T6 breadth]."""
    cache_dir = tmp_path / "cache"
    uniprot_id = "P12345"

    with patch("aminx.io.proxide_fetch.Path.mkdir", side_effect=OSError("No space left")):
      with pytest.raises(InputResolutionError) as exc_info:
        fetch_afdb(uniprot_id, cache_dir)
      assert "cannot write input cache" in str(exc_info.value)
      assert "OSError" in str(exc_info.value)

  def test_full_disk_mdcath_error(self, tmp_path: Path) -> None:
    """m2: OSError (full disk) on mdcath -> InputResolutionError [T6 breadth]."""
    cache_dir = tmp_path / "cache"
    md_cath_id = "1abcA00"

    with patch("aminx.io.proxide_fetch.Path.mkdir", side_effect=OSError("No space left")):
      with pytest.raises(InputResolutionError) as exc_info:
        fetch_md_cath(md_cath_id, cache_dir)
      assert "cannot write input cache" in str(exc_info.value)
      assert "OSError" in str(exc_info.value)


class TestInputResolutionError:
  """Tests for InputResolutionError exception class."""

  def test_input_resolution_error_is_exception(self) -> None:
    """InputResolutionError is an Exception subclass."""
    assert issubclass(InputResolutionError, Exception)

  def test_input_resolution_error_can_be_raised(self) -> None:
    """InputResolutionError can be raised and caught."""
    with pytest.raises(InputResolutionError) as exc_info:
      raise InputResolutionError("test message")
    assert "test message" in str(exc_info.value)
