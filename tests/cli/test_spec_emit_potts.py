"""Tests for ``aminx spec emit-potts`` CLI subcommand.

Verifies that the emit-potts command emits valid PottsRunSpec JSON with
customizable k_neighbors, n_backbones, and weights_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aminx.cli import app
from aminx.potts.spec import PottsRunSpec

runner = CliRunner()


class TestEmitPotts:
    """Tests for ``aminx spec emit-potts``."""

    def test_emit_potts_default(self) -> None:
        """emit-potts with defaults exits 0 and produces valid JSON.

        Default values: k_neighbors=48, n_backbones=1, weights_path="".
        """
        result = runner.invoke(app, ["spec", "emit-potts"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["k_neighbors"] == 48
        assert parsed["n_backbones"] == 1
        assert parsed["weights_path"] == ""

    def test_emit_potts_custom_args(self) -> None:
        """emit-potts with custom --k-neighbors and --n-backbones."""
        result = runner.invoke(
            app,
            [
                "spec",
                "emit-potts",
                "--k-neighbors",
                "32",
                "--n-backbones",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["k_neighbors"] == 32
        assert parsed["n_backbones"] == 4

    def test_emit_potts_custom_weights_path(self) -> None:
        """emit-potts with custom --weights-path."""
        result = runner.invoke(
            app,
            [
                "spec",
                "emit-potts",
                "--weights-path",
                "/path/to/weights.eqx",
                "--k-neighbors",
                "48",
            ],
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["weights_path"] == "/path/to/weights.eqx"

    def test_emit_potts_round_trips(self) -> None:
        """emit-potts output can be loaded back via PottsRunSpec.from_json."""
        result = runner.invoke(
            app,
            [
                "spec",
                "emit-potts",
                "--k-neighbors",
                "48",
                "--n-backbones",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        # Verify the JSON can be round-tripped
        spec = PottsRunSpec.from_json(result.output)
        assert spec.k_neighbors == 48
        assert spec.n_backbones == 2

    def test_emit_potts_compact_flag(self) -> None:
        """--compact produces single-line JSON."""
        result = runner.invoke(
            app,
            ["spec", "emit-potts", "--compact"],
        )
        assert result.exit_code == 0, result.output
        json_text = result.output.strip()
        assert "\n" not in json_text
        # must still be valid JSON
        json.loads(json_text)

    def test_emit_potts_out_flag_writes_file(self, tmp_path: Path) -> None:
        """--out writes JSON to file and exits 0."""
        out_path = tmp_path / "potts_spec.json"
        result = runner.invoke(
            app,
            [
                "spec",
                "emit-potts",
                "--k-neighbors",
                "48",
                "--out",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        parsed = json.loads(out_path.read_text(encoding="utf-8"))
        assert parsed["k_neighbors"] == 48
        assert parsed["n_backbones"] == 1
