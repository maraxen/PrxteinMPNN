"""Tests for the aminx Typer CLI (src/aminx/cli.py).

Covers all three sub-commands of ``aminx spec``:
  - ``validate``        — reads JSON, validates spec class, exits 0/1
  - ``roundtrip``       — re-serializes spec, writes to --out or stdout
  - ``portable-roundtrip`` — processes the portable RunSpec subset

Uses ``typer.testing.CliRunner`` for in-process invocation — no subprocess,
no JAX, no model weights, no GPU required.  All specs are constructed from
pure Python/JSON values (``inputs`` is a plain string path, never loaded).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aminx.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers: minimal valid JSON blobs
# ---------------------------------------------------------------------------

def _sampling_spec_json() -> dict:
    """Minimal SamplingSpecification JSON dict (all required fields present)."""
    return {
        "_spec_class": "SamplingSpecification",
        "inputs": "test.pdb",
    }


def _scoring_spec_json() -> dict:
    """Minimal ScoringSpecification JSON dict."""
    return {
        "_spec_class": "ScoringSpecification",
        "inputs": "test.pdb",
        "sequences_to_score": ["ACDEFGHIKLMNPQRSTVWY"],
    }


def _portable_spec_dict() -> dict:
    """Minimal portable RunSpec dict (v2 wire format)."""
    return {
        "version": 2,
        "io": {
            "sink_kind": "none",
            "output_dir": None,
            "manifest_path": None,
        },
        "multistate": {
            "mode": "single",
            "n_states": 1,
            "combine_strategy": "mean",
        },
        "resource": {
            "n_devices": 1,
            "sample_batch_size": 1,
            "structure_batch_size": 1,
            "max_buffer_size": None,
        },
        "precision": {"compute": "fp32"},
    }


def _write_json(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


# ===========================================================================
# validate command
# ===========================================================================


class TestValidate:
    """Tests for ``aminx spec validate <path>``."""

    @pytest.mark.parametrize(
        "spec_fn,expected_class",
        [
            (_sampling_spec_json, "SamplingSpecification"),
            (_scoring_spec_json, "ScoringSpecification"),
        ],
        ids=["sampling", "scoring"],
    )
    def test_valid_spec_exits_zero(self, tmp_path: Path, spec_fn, expected_class: str) -> None:
        """Valid spec JSON exits 0 and prints 'OK: <ClassName>'."""
        path = _write_json(tmp_path, "spec.json", spec_fn())
        result = runner.invoke(app, ["spec", "validate", str(path)])
        assert result.exit_code == 0, result.output
        assert f"OK: {expected_class}" in result.output

    def test_malformed_json_exits_nonzero(self, tmp_path: Path) -> None:
        """A file containing invalid JSON syntax must exit non-zero."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json!!!", encoding="utf-8")
        result = runner.invoke(app, ["spec", "validate", str(bad)])
        assert result.exit_code != 0

    def test_missing_spec_class_key_exits_nonzero(self, tmp_path: Path) -> None:
        """Valid JSON that is missing the '_spec_class' discriminator exits non-zero."""
        payload = {"inputs": "test.pdb"}  # no _spec_class
        path = _write_json(tmp_path, "no_class.json", payload)
        result = runner.invoke(app, ["spec", "validate", str(path)])
        assert result.exit_code != 0

    def test_empty_object_exits_nonzero(self, tmp_path: Path) -> None:
        """An empty JSON object {} exits non-zero (missing required fields)."""
        path = _write_json(tmp_path, "empty.json", {})
        result = runner.invoke(app, ["spec", "validate", str(path)])
        assert result.exit_code != 0

    def test_nonexistent_file_exits_nonzero(self, tmp_path: Path) -> None:
        """A path that does not exist exits non-zero (Typer's exists=True guard)."""
        missing = tmp_path / "ghost.json"
        result = runner.invoke(app, ["spec", "validate", str(missing)])
        assert result.exit_code != 0

    def test_valid_spec_json_contains_class_name_only_once(self, tmp_path: Path) -> None:
        """Output for valid spec prints exactly one 'OK:' line."""
        path = _write_json(tmp_path, "spec.json", _sampling_spec_json())
        result = runner.invoke(app, ["spec", "validate", str(path)])
        assert result.exit_code == 0, result.output
        ok_lines = [ln for ln in result.output.splitlines() if ln.startswith("OK:")]
        assert len(ok_lines) == 1


# ===========================================================================
# roundtrip command
# ===========================================================================


class TestRoundtrip:
    """Tests for ``aminx spec roundtrip <path> [--out <path>] [--compact]``."""

    def test_valid_spec_stdout_exits_zero(self, tmp_path: Path) -> None:
        """Valid spec JSON printed to stdout, exit 0, and is parseable JSON."""
        path = _write_json(tmp_path, "spec.json", _sampling_spec_json())
        result = runner.invoke(app, ["spec", "roundtrip", str(path)])
        assert result.exit_code == 0, result.output
        # Output must be parseable JSON
        parsed = json.loads(result.output)
        assert parsed["_spec_class"] == "SamplingSpecification"

    def test_out_flag_writes_file(self, tmp_path: Path) -> None:
        """--out writes re-serialized JSON to the given path and prints 'Wrote'."""
        spec_path = _write_json(tmp_path, "spec.json", _sampling_spec_json())
        out_path = tmp_path / "output.json"
        result = runner.invoke(app, ["spec", "roundtrip", str(spec_path), "--out", str(out_path)])
        assert result.exit_code == 0, result.output
        assert "Wrote" in result.output
        assert str(out_path) in result.output
        assert out_path.exists()
        written = json.loads(out_path.read_text(encoding="utf-8"))
        assert written["_spec_class"] == "SamplingSpecification"

    def test_compact_flag_produces_single_line(self, tmp_path: Path) -> None:
        """--compact produces JSON with no internal newlines in the spec body."""
        path = _write_json(tmp_path, "spec.json", _sampling_spec_json())
        result = runner.invoke(app, ["spec", "roundtrip", str(path), "--compact"])
        assert result.exit_code == 0, result.output
        # Strip trailing newline added by echo, then the JSON itself must be one line
        json_text = result.output.strip()
        assert "\n" not in json_text

    def test_invalid_spec_json_exits_nonzero(self, tmp_path: Path) -> None:
        """A file with invalid JSON (syntax error) must exit non-zero."""
        bad = tmp_path / "bad.json"
        bad.write_text("{ invalid", encoding="utf-8")
        result = runner.invoke(app, ["spec", "roundtrip", str(bad)])
        assert result.exit_code != 0

    def test_roundtrip_is_idempotent(self, tmp_path: Path) -> None:
        """Applying roundtrip twice yields the same JSON document."""
        spec_path = _write_json(tmp_path, "spec.json", _sampling_spec_json())
        # First pass — stdout
        r1 = runner.invoke(app, ["spec", "roundtrip", str(spec_path)])
        assert r1.exit_code == 0, r1.output
        # Write first-pass output to a new file
        pass1_path = tmp_path / "pass1.json"
        pass1_path.write_text(r1.output, encoding="utf-8")
        # Second pass
        r2 = runner.invoke(app, ["spec", "roundtrip", str(pass1_path)])
        assert r2.exit_code == 0, r2.output
        # Both outputs must parse to equal dicts
        assert json.loads(r1.output) == json.loads(r2.output)


# ===========================================================================
# portable-roundtrip command
# ===========================================================================


class TestPortableRoundtrip:
    """Tests for ``aminx spec portable-roundtrip <path> [--compact]``."""

    def test_valid_portable_dict_exits_zero(self, tmp_path: Path) -> None:
        """Valid portable spec dict exits 0 and output is parseable JSON."""
        path = _write_json(tmp_path, "portable.json", _portable_spec_dict())
        result = runner.invoke(app, ["spec", "portable-roundtrip", str(path)])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)

    def test_list_top_level_exits_nonzero(self, tmp_path: Path) -> None:
        """Top-level JSON list (not dict) exits 1 with an error about 'object'."""
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = runner.invoke(app, ["spec", "portable-roundtrip", str(path)])
        assert result.exit_code == 1
        assert "object" in result.output.lower()

    def test_compact_flag_produces_single_line(self, tmp_path: Path) -> None:
        """--compact produces single-line JSON output."""
        path = _write_json(tmp_path, "portable.json", _portable_spec_dict())
        result = runner.invoke(app, ["spec", "portable-roundtrip", str(path), "--compact"])
        assert result.exit_code == 0, result.output
        json_text = result.output.strip()
        assert "\n" not in json_text

    def test_invalid_portable_content_exits_nonzero(self, tmp_path: Path) -> None:
        """A portable dict missing required keys (e.g. 'version') exits non-zero."""
        bad = {"multistate": {"mode": "single", "n_states": 1, "combine_strategy": "mean"}}
        path = _write_json(tmp_path, "bad_portable.json", bad)
        result = runner.invoke(app, ["spec", "portable-roundtrip", str(path)])
        assert result.exit_code != 0

    def test_portable_roundtrip_output_contains_version(self, tmp_path: Path) -> None:
        """Output dict includes a 'version' key matching the current wire version."""
        path = _write_json(tmp_path, "portable.json", _portable_spec_dict())
        result = runner.invoke(app, ["spec", "portable-roundtrip", str(path)])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert "version" in parsed
        assert isinstance(parsed["version"], int)

    def test_portable_roundtrip_output_contains_expected_keys(self, tmp_path: Path) -> None:
        """Output dict contains the expected top-level sections."""
        path = _write_json(tmp_path, "portable.json", _portable_spec_dict())
        result = runner.invoke(app, ["spec", "portable-roundtrip", str(path)])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        for key in ("multistate", "resource", "precision"):
            assert key in parsed, f"Missing key {key!r} in portable output"

    def test_malformed_json_exits_nonzero(self, tmp_path: Path) -> None:
        """A file with invalid JSON syntax exits non-zero."""
        bad = tmp_path / "bad.json"
        bad.write_text("{ broken }", encoding="utf-8")
        result = runner.invoke(app, ["spec", "portable-roundtrip", str(bad)])
        assert result.exit_code != 0

    def test_v1_portable_dict_supported(self, tmp_path: Path) -> None:
        """v1 portable spec (no 'io' key) is also accepted by the codec."""
        v1 = {
            "version": 1,
            "multistate": {"mode": "single", "n_states": 1, "combine_strategy": "mean"},
            "resource": {
                "n_devices": 1,
                "sample_batch_size": 1,
                "structure_batch_size": 1,
                "max_buffer_size": None,
            },
            "precision": {"compute": "fp32"},
        }
        path = _write_json(tmp_path, "v1_portable.json", v1)
        result = runner.invoke(app, ["spec", "portable-roundtrip", str(path)])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)
