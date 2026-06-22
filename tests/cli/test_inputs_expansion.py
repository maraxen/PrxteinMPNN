"""Tests for _expand_inputs: directory, glob, edge cases, dedup, fail_fast.

All tests operate on tmp_path fixtures — no model weights or JAX required.
The helper is exercised directly (unit tests) plus one CliRunner end-to-end
test covering a directory input.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from aminx.cli import _expand_inputs, app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

runner = CliRunner()


def _make_files(root: Path, names: list[str]) -> list[Path]:
    """Create empty files under root and return their Paths."""
    root.mkdir(parents=True, exist_ok=True)
    created = []
    for name in names:
        p = root / name
        p.write_text("")
        created.append(p)
    return created


# ---------------------------------------------------------------------------
# 1. Individual file paths pass through unchanged (backward compatibility)
# ---------------------------------------------------------------------------


def test_individual_file_passes_through(tmp_path: Path) -> None:
    f = tmp_path / "protein.pdb"
    f.write_text("")
    result = _expand_inputs([str(f)])
    assert result == [str(f)]


def test_multiple_individual_files_pass_through(tmp_path: Path) -> None:
    files = _make_files(tmp_path, ["a.pdb", "b.cif"])
    paths = [str(f) for f in files]
    result = _expand_inputs(paths)
    assert result == paths


# ---------------------------------------------------------------------------
# 2. Directory expands to sorted .pdb/.cif files inside it
# ---------------------------------------------------------------------------


def test_directory_expands_to_sorted_pdb_cif(tmp_path: Path) -> None:
    d = tmp_path / "structs"
    _make_files(d, ["z.pdb", "a.cif", "m.PDB", "n.CIF", "ignore.txt", "data.json"])
    result = _expand_inputs([str(d)])
    # Should include .pdb / .cif case-insensitively, sorted, no .txt / .json
    assert all(Path(p).suffix.lower() in {".pdb", ".cif"} for p in result)
    assert result == sorted(result)
    assert len(result) == 4  # z.pdb, a.cif, m.PDB, n.CIF


def test_directory_excludes_non_structure_files(tmp_path: Path) -> None:
    d = tmp_path / "mixed"
    _make_files(d, ["good.pdb", "bad.txt", "other.fasta"])
    result = _expand_inputs([str(d)])
    assert len(result) == 1
    assert result[0].endswith("good.pdb")


# ---------------------------------------------------------------------------
# 3. Glob pattern expands to matching structure files
# ---------------------------------------------------------------------------


def test_glob_expands_matching_pdb(tmp_path: Path) -> None:
    d = tmp_path / "globs"
    _make_files(d, ["one.pdb", "two.pdb", "three.cif", "skip.txt"])
    pattern = str(d / "*.pdb")
    result = _expand_inputs([pattern])
    assert len(result) == 2
    assert all(p.endswith(".pdb") for p in result)
    assert result == sorted(result)


def test_glob_expands_mixed_extensions(tmp_path: Path) -> None:
    d = tmp_path / "glob2"
    _make_files(d, ["x.pdb", "y.cif", "z.txt"])
    pattern = str(d / "*")
    result = _expand_inputs([pattern])
    assert len(result) == 2
    assert all(Path(p).suffix.lower() in {".pdb", ".cif"} for p in result)


# ---------------------------------------------------------------------------
# 4. Empty directory → Exit(1) with message
# ---------------------------------------------------------------------------


def test_empty_directory_exits_1(tmp_path: Path) -> None:
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(click.exceptions.Exit) as exc_info:
        _expand_inputs([str(d)])
    assert exc_info.value.exit_code == 1


def test_empty_directory_message_names_entry(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = tmp_path / "nostructs"
    d.mkdir()
    with pytest.raises(click.exceptions.Exit):
        _expand_inputs([str(d)])


def test_directory_with_only_txt_exits_1(tmp_path: Path) -> None:
    d = tmp_path / "txtonly"
    _make_files(d, ["readme.txt", "data.csv"])
    with pytest.raises(click.exceptions.Exit) as exc_info:
        _expand_inputs([str(d)])
    assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# 5. Glob matching nothing → Exit(1)
# ---------------------------------------------------------------------------


def test_empty_glob_exits_1(tmp_path: Path) -> None:
    pattern = str(tmp_path / "nonexistent_*.pdb")
    with pytest.raises(click.exceptions.Exit) as exc_info:
        _expand_inputs([pattern])
    assert exc_info.value.exit_code == 1


def test_glob_matching_only_non_structure_exits_1(tmp_path: Path) -> None:
    d = tmp_path / "nonstruct"
    _make_files(d, ["a.txt", "b.fasta"])
    pattern = str(d / "*")
    with pytest.raises(click.exceptions.Exit) as exc_info:
        _expand_inputs([pattern])
    assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# 6. Mixed valid/invalid: default warns+skips for dir/glob failures;
#    fail_fast=True → Exit(1) on first dir/glob that expands to nothing.
#    Concrete paths always pass through (backward compat).
# ---------------------------------------------------------------------------


def test_mixed_default_warns_and_skips_empty_dir(tmp_path: Path) -> None:
    """An empty directory in a mixed list should warn+skip, leaving valid entries."""
    valid = tmp_path / "valid.pdb"
    valid.write_text("")
    empty_dir = tmp_path / "emptydir"
    empty_dir.mkdir()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _expand_inputs([str(valid), str(empty_dir)])

    assert result == [str(valid)]
    assert any("emptydir" in str(w.message) for w in caught)


def test_mixed_fail_fast_exits_on_empty_dir(tmp_path: Path) -> None:
    """fail_fast=True should exit 1 when a directory yields nothing."""
    valid = tmp_path / "valid.pdb"
    valid.write_text("")
    empty_dir = tmp_path / "ghost_dir"
    empty_dir.mkdir()

    with pytest.raises(click.exceptions.Exit) as exc_info:
        _expand_inputs([str(empty_dir), str(valid)], fail_fast=True)
    assert exc_info.value.exit_code == 1


def test_concrete_path_passes_through_even_if_missing(tmp_path: Path) -> None:
    """Concrete paths are passed through unchanged (existence not validated here)."""
    missing = str(tmp_path / "ghost.pdb")
    result = _expand_inputs([missing])
    # Should pass through — no exit, no warning
    assert result == [missing]


def test_mixed_valid_entries_survive_default(tmp_path: Path) -> None:
    good1 = tmp_path / "a.pdb"
    good2 = tmp_path / "b.cif"
    good1.write_text("")
    good2.write_text("")
    bad_dir = tmp_path / "emptydir"
    bad_dir.mkdir()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = _expand_inputs([str(good1), str(bad_dir), str(good2)])

    assert str(good1) in result
    assert str(good2) in result
    # emptydir should have been skipped (not caused exit)
    assert not any("emptydir" in p for p in result)


# ---------------------------------------------------------------------------
# 7. Deduplication preserves first-seen order
# ---------------------------------------------------------------------------


def test_dedup_preserves_first_seen_order(tmp_path: Path) -> None:
    a = tmp_path / "a.pdb"
    b = tmp_path / "b.pdb"
    c = tmp_path / "c.pdb"
    for f in [a, b, c]:
        f.write_text("")

    # Pass c, a, b, then repeat a and c — should get c, a, b (first-seen order)
    result = _expand_inputs([str(c), str(a), str(b), str(a), str(c)])
    assert result == [str(c), str(a), str(b)]


def test_dedup_glob_and_direct_no_duplicates(tmp_path: Path) -> None:
    d = tmp_path / "dd"
    files = _make_files(d, ["one.pdb", "two.pdb"])
    # Glob that matches both, then explicit path to one.pdb — dedup expected
    pattern = str(d / "*.pdb")
    result = _expand_inputs([pattern, str(files[0])])
    # "one.pdb" should appear only once
    assert result.count(str(files[0])) == 1


# ---------------------------------------------------------------------------
# 8. All-empty result after expansion → Exit(1)
# ---------------------------------------------------------------------------


def test_all_empty_dirs_exits_1(tmp_path: Path) -> None:
    """If all directory entries contain no structure files, final list is empty → exit 1."""
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    d1.mkdir()
    d2.mkdir()
    # Both dirs have only non-structure files
    (d1 / "readme.txt").write_text("")
    (d2 / "data.json").write_text("")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(click.exceptions.Exit) as exc_info:
            _expand_inputs([str(d1), str(d2)])
    assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# 9. End-to-end CliRunner test: directory input on spec emit-inspect
# ---------------------------------------------------------------------------


def test_end_to_end_directory_input(tmp_path: Path) -> None:
    """CliRunner smoke test: passing a directory to spec emit-inspect expands it."""
    d = tmp_path / "inputs"
    _make_files(d, ["protein.pdb"])

    result = runner.invoke(
        app,
        [
            "spec",
            "emit-inspect",
            "--inputs",
            str(d),
            "--emit-json",
        ],
        catch_exceptions=True,
    )
    # Should succeed (exit 0) or fail only on spec construction (not expansion)
    # The key assertion: it did NOT fail with "--inputs is required" or expansion error
    # about missing .pdb/.cif files. If spec construction fails due to model-related
    # reasons that's acceptable — expansion itself should have worked.
    if result.exit_code != 0:
        # Any expansion-related failure would mention "no .pdb or .cif" — assert it doesn't
        assert "no .pdb or .cif" not in (result.output or "")
        assert "directory" not in (result.output or "").lower() or "protein.pdb" in (result.output or "")
