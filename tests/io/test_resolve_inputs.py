"""Tests for resolve_inputs resolver core (URI dispatch + local expansion + dedup).

Tests cover:
- Local path/dir/glob resolution (regression against _expand_inputs)
- Remote URI dispatch (pdb://, afdb://, mdcath://)
- Pre-fetch deduplication (entry-level, before any fetch)
- Cache dir precedence (arg -> env -> XDG)
- input_type flag behavior (auto, file, pdb/afdb/mdcath)
- fail_fast semantics (True re-raises, False warns+skips)
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from aminx.cli import _resolve_cache_dir, resolve_inputs
from aminx.io.proxide_fetch import InputResolutionError


class TestResolveCacheDir:
  """Tests for cache dir resolution with precedence."""

  def test_explicit_cache_dir_takes_priority(self, tmp_path: Path) -> None:
    """Explicit cache_dir arg overrides env and defaults."""
    with patch.dict(os.environ, {"AMINX_CACHE_DIR": str(tmp_path / "env")}):
      result = _resolve_cache_dir(str(tmp_path / "arg"))
      assert result == tmp_path / "arg"

  def test_aminx_cache_dir_env_secondpriority(self, tmp_path: Path) -> None:
    """AMINX_CACHE_DIR env var used when no explicit arg."""
    env_path = tmp_path / "env_cache"
    with patch.dict(os.environ, {"AMINX_CACHE_DIR": str(env_path)}, clear=False):
      result = _resolve_cache_dir(None)
      assert result == env_path

  def test_xdg_cache_home_respected(self, tmp_path: Path) -> None:
    """XDG_CACHE_HOME env var respected in default."""
    xdg_path = tmp_path / "xdg"
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(xdg_path)}, clear=False):
      # Clear AMINX_CACHE_DIR to test XDG fallback
      with patch.dict(os.environ, {"AMINX_CACHE_DIR": ""}, clear=False):
        result = _resolve_cache_dir(None)
        assert result == xdg_path / "aminx" / "inputs"

  def test_default_xdg_home_cache(self) -> None:
    """Default falls back to ~/.cache/aminx/inputs."""
    with patch.dict(
      os.environ, {"AMINX_CACHE_DIR": "", "XDG_CACHE_HOME": ""}, clear=False
    ):
      result = _resolve_cache_dir(None)
      assert result == Path.home() / ".cache" / "aminx" / "inputs"


class TestResolveInputsLocalRegression:
  """Regression tests: local paths resolve identically to _expand_inputs behavior."""

  def test_concrete_file_passthrough(self, tmp_path: Path) -> None:
    """Concrete file path passed through unchanged."""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text("dummy")
    result = resolve_inputs([str(pdb_file)], input_type="file", cache_dir=tmp_path)
    assert result == [str(pdb_file)]

  def test_directory_expansion(self, tmp_path: Path) -> None:
    """Directory expands to all .pdb/.cif files inside (sorted, non-recursive)."""
    (tmp_path / "a.pdb").write_text("a")
    (tmp_path / "c.cif").write_text("c")
    (tmp_path / "b.txt").write_text("b")  # Filtered out
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "d.pdb").write_text("d")  # Not included (non-recursive)

    result = resolve_inputs([str(tmp_path)], input_type="file", cache_dir=tmp_path)
    # Should be sorted by path
    expected_paths = sorted([str(tmp_path / "a.pdb"), str(tmp_path / "c.cif")])
    assert result == expected_paths

  def test_glob_expansion(self, tmp_path: Path) -> None:
    """Glob pattern expands to matching structure files."""
    (tmp_path / "1ubq.pdb").write_text("1ubq")
    (tmp_path / "2xyz.pdb").write_text("2xyz")
    (tmp_path / "3abc.cif").write_text("3abc")

    result = resolve_inputs(
      [str(tmp_path / "*.pdb")], input_type="file", cache_dir=tmp_path
    )
    expected = sorted([str(tmp_path / "1ubq.pdb"), str(tmp_path / "2xyz.pdb")])
    assert result == expected

  def test_dedup_preserves_firstseen(self, tmp_path: Path) -> None:
    """Dedup preserves first-seen order (no sort)."""
    pdb = tmp_path / "test.pdb"
    pdb.write_text("test")
    result = resolve_inputs([str(pdb), str(pdb)], input_type="file", cache_dir=tmp_path)
    assert result == [str(pdb)]

  def test_empty_directory_fail_fast_true(self, tmp_path: Path) -> None:
    """Empty directory with fail_fast=True exits 1."""
    with pytest.raises(typer.Exit) as exc_info:
      resolve_inputs([str(tmp_path)], input_type="file", cache_dir=tmp_path, fail_fast=True)
    assert exc_info.value.exit_code == 1

  def test_empty_directory_fail_fast_false(self, tmp_path: Path) -> None:
    """Empty directory with fail_fast=False warns and skips."""
    # This will fail because no valid files are found overall
    with pytest.raises(typer.Exit) as exc_info:
      resolve_inputs(
        [str(tmp_path)], input_type="file", cache_dir=tmp_path, fail_fast=False
      )
    assert exc_info.value.exit_code == 1

  def test_glob_no_matches_fail_fast_false(self, tmp_path: Path) -> None:
    """Glob with no matches and fail_fast=False warns and skips."""
    (tmp_path / "dummy.txt").write_text("dummy")
    with pytest.raises(typer.Exit) as exc_info:
      resolve_inputs(
        [str(tmp_path / "*.pdb")], input_type="file", cache_dir=tmp_path, fail_fast=False
      )
    assert exc_info.value.exit_code == 1


class TestResolveInputsRemoteDispatch:
  """Tests for remote URI dispatch (pdb://, afdb://, mdcath://)."""

  def test_pdb_uri_dispatches_to_fetch_pdb(self, tmp_path: Path) -> None:
    """pdb://ID dispatches to fetch_pdb."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1ubq.cif"
      result = resolve_inputs(
        ["pdb://1ubq"],
        input_type="auto",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once_with("1ubq", tmp_path, fmt="mmcif")
      assert result == [str(tmp_path / "1ubq.cif")]

  def test_pdb_with_format_suffix(self, tmp_path: Path) -> None:
    """pdb://ID.pdb uses format_type='pdb'."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1ubq.pdb"
      result = resolve_inputs(
        ["pdb://1ubq.pdb"],
        input_type="auto",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once_with("1ubq", tmp_path, fmt="pdb")
      assert result == [str(tmp_path / "1ubq.pdb")]

  def test_afdb_uri_dispatches_to_fetch_afdb(self, tmp_path: Path) -> None:
    """afdb://ID dispatches to fetch_afdb."""
    with patch("aminx.cli.fetch_afdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "P12345.cif"
      result = resolve_inputs(
        ["afdb://P12345"],
        input_type="auto",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once_with("P12345", tmp_path)
      assert result == [str(tmp_path / "P12345.cif")]

  def test_mdcath_uri_dispatches_to_fetch_md_cath(self, tmp_path: Path) -> None:
    """mdcath://ID dispatches to fetch_md_cath."""
    with patch("aminx.cli.fetch_md_cath") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1abcA00.h5"
      result = resolve_inputs(
        ["mdcath://1abcA00"],
        input_type="auto",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once_with("1abcA00", tmp_path)
      assert result == [str(tmp_path / "1abcA00.h5")]


class TestResolveInputsPreFetchDedup:
  """Tests for pre-fetch deduplication."""

  def test_prededup_identical_pdb_uris(self, tmp_path: Path) -> None:
    """Identical pdb:// URIs deduped before fetch; fetch called once."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1ubq.cif"
      result = resolve_inputs(
        ["pdb://1ubq", "pdb://1ubq"],
        input_type="auto",
        cache_dir=tmp_path,
      )
      # Fetch should be called only once (pre-dedup)
      assert mock_fetch.call_count == 1
      assert result == [str(tmp_path / "1ubq.cif")]

  def test_prededup_different_formats_not_deduped(self, tmp_path: Path) -> None:
    """pdb://1ubq vs pdb://1ubq.pdb are distinct (different fmt) and both fetched."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.side_effect = [
        tmp_path / "1ubq_mmcif.cif",
        tmp_path / "1ubq_pdb.pdb",
      ]
      result = resolve_inputs(
        ["pdb://1ubq", "pdb://1ubq.pdb"],
        input_type="auto",
        cache_dir=tmp_path,
      )
      # Both should be fetched (different formats)
      assert mock_fetch.call_count == 2
      assert len(result) == 2


class TestResolveInputsInputTypeFlag:
  """Tests for input_type flag behavior."""

  def test_input_type_file_suppresses_scheme_detection(self, tmp_path: Path) -> None:
    """input_type='file' treats pdb://1ubq as a literal local path, no fetcher."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      # pdb://1ubq should be treated as a literal filename (a concrete path)
      # It gets passed through even if it doesn't exist (downstream runner validates)
      result = resolve_inputs(
        ["pdb://1ubq"],
        input_type="file",
        cache_dir=tmp_path,
      )
      # Should return the literal string unchanged
      assert result == ["pdb://1ubq"]
      # Fetch should NOT be called
      mock_fetch.assert_not_called()

  def test_input_type_pdb_applies_to_schemeless(self, tmp_path: Path) -> None:
    """input_type='pdb' treats bare tokens as pdb:// IDs."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.return_value = tmp_path / "1ubq.cif"
      result = resolve_inputs(
        ["1ubq"],
        input_type="pdb",
        cache_dir=tmp_path,
      )
      mock_fetch.assert_called_once_with("1ubq", tmp_path, fmt="mmcif")
      assert result == [str(tmp_path / "1ubq.cif")]

  def test_input_type_pdb_with_afdb_uri_raises(self, tmp_path: Path) -> None:
    """input_type='pdb' with explicit afdb:// URI raises BadParameter."""
    with pytest.raises(typer.BadParameter) as exc_info:
      resolve_inputs(
        ["afdb://P12345"],
        input_type="pdb",
        cache_dir=tmp_path,
      )
    assert "conflicts with explicit scheme" in str(exc_info.value)


class TestResolveInputsFailFast:
  """Tests for fail_fast semantics."""

  def test_fail_fast_true_reraises_fetch_error(self, tmp_path: Path) -> None:
    """fail_fast=True re-raises InputResolutionError from fetch."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.side_effect = InputResolutionError("Network error")
      with pytest.raises(InputResolutionError):
        resolve_inputs(
          ["pdb://1ubq"],
          input_type="auto",
          cache_dir=tmp_path,
          fail_fast=True,
        )

  def test_fail_fast_false_warns_and_skips_fetch_error(self, tmp_path: Path) -> None:
    """fail_fast=False warns and skips on fetch error, but final empty list fails."""
    with patch("aminx.cli.fetch_pdb") as mock_fetch:
      mock_fetch.side_effect = InputResolutionError("Network error")
      with pytest.raises(typer.Exit) as exc_info:
        resolve_inputs(
          ["pdb://1ubq"],
          input_type="auto",
          cache_dir=tmp_path,
          fail_fast=False,
        )
      # Should exit because no valid results
      assert exc_info.value.exit_code == 1


class TestResolveInputsMixedInputs:
  """Tests for mixed local and remote inputs."""

  def test_mixed_local_and_remote(self, tmp_path: Path) -> None:
    """Mixed --inputs pdb://1ubq /tmp/x.pdb afdb://P1 resolves independently."""
    local_file = tmp_path / "local.pdb"
    local_file.write_text("local")

    with patch("aminx.cli.fetch_pdb") as mock_pdb, patch(
      "aminx.cli.fetch_afdb"
    ) as mock_afdb:
      mock_pdb.return_value = tmp_path / "1ubq.cif"
      mock_afdb.return_value = tmp_path / "P12345.cif"

      result = resolve_inputs(
        ["pdb://1ubq", str(local_file), "afdb://P12345"],
        input_type="auto",
        cache_dir=tmp_path,
      )

      assert mock_pdb.call_count == 1
      assert mock_afdb.call_count == 1
      # Result should contain all three (in order they were resolved)
      assert len(result) == 3


class TestResolveInputsEmptyInputs:
  """Tests for empty input handling."""

  def test_empty_inputs_raises(self, tmp_path: Path) -> None:
    """Empty input list raises Exit(1)."""
    with pytest.raises(typer.Exit) as exc_info:
      resolve_inputs([], input_type="auto", cache_dir=tmp_path)
    assert exc_info.value.exit_code == 1


class TestResolveInputsFileScheme:
  """Tests for file:// scheme handling."""

  def test_file_uri_strips_to_local_path(self, tmp_path: Path) -> None:
    """file:///path resolves to local path."""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text("test")
    result = resolve_inputs(
      [f"file://{str(pdb_file)}"],
      input_type="auto",
      cache_dir=tmp_path,
    )
    assert result == [str(pdb_file)]

  def test_file_uri_with_host_strips_host(self, tmp_path: Path) -> None:
    """file://localhost/path strips host and resolves /path."""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text("test")
    # file://localhost/absolute/path becomes /absolute/path
    result = resolve_inputs(
      [f"file://localhost{str(pdb_file)}"],
      input_type="auto",
      cache_dir=tmp_path,
    )
    assert result == [str(pdb_file)]
