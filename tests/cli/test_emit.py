"""Equivalence tests: ``aminx spec emit-*`` must produce identical JSON to ``aminx run * --emit-json``.

For each spec type (sample, score, jacobian, inspect), we verify:
  - ``aminx spec emit-<type>`` exits 0 and produces valid JSON
  - ``aminx run <type> --emit-json`` exits 0 and produces valid JSON
  - Both JSON dicts are identical (both use sort_keys=True)

Also tests --compact (single-line JSON) and --out (file write) for emit-sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aminx.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Equivalence parametrized test (required by spec)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec_type,extra_args,expected_class",
    [
        ("sample", [], "SamplingSpecification"),
        ("score", ["--sequences-to-score", "ACDE"], "ScoringSpecification"),
        ("jacobian", [], "JacobianSpecification"),
        ("inspect", [], "InspectionSpecification"),
    ],
)
def test_emit_equivalence(
    tmp_path: Path,
    spec_type: str,
    extra_args: list[str],
    expected_class: str,
) -> None:
    """emit-* and run * --emit-json produce identical JSON dicts."""
    inputs = ["--inputs", "fake.pdb"]

    # emit path: aminx spec emit-<type> --inputs fake.pdb [extra_args...]
    emit_result = runner.invoke(app, ["spec", f"emit-{spec_type}"] + inputs + extra_args)
    assert emit_result.exit_code == 0, (
        f"spec emit-{spec_type} failed (exit {emit_result.exit_code}):\n{emit_result.output}"
    )

    # run --emit-json path: aminx run <type> --inputs fake.pdb [extra_args...] --emit-json
    run_result = runner.invoke(app, ["run", spec_type] + inputs + extra_args + ["--emit-json"])
    assert run_result.exit_code == 0, (
        f"run {spec_type} --emit-json failed (exit {run_result.exit_code}):\n{run_result.output}"
    )

    # must produce identical dicts (modulo key ordering — both use sort_keys=True)
    emit_dict = json.loads(emit_result.output)
    run_dict = json.loads(run_result.output)
    assert emit_dict == run_dict, (
        f"emit-{spec_type} and run {spec_type} --emit-json produced different JSON.\n"
        f"emit keys: {sorted(emit_dict)}\nrun keys: {sorted(run_dict)}"
    )

    # verify the expected spec class is present
    assert emit_dict.get("_spec_class") == expected_class


# ---------------------------------------------------------------------------
# emit-sample additional tests
# ---------------------------------------------------------------------------


class TestEmitSample:
    """Tests for ``aminx spec emit-sample``."""

    def test_exits_zero_produces_valid_json(self) -> None:
        """emit-sample with minimal args exits 0 and emits valid JSON."""
        result = runner.invoke(app, ["spec", "emit-sample", "--inputs", "test.pdb"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["_spec_class"] == "SamplingSpecification"

    def test_compact_flag_produces_single_line(self) -> None:
        """--compact produces single-line JSON with no internal newlines."""
        result = runner.invoke(
            app, ["spec", "emit-sample", "--inputs", "test.pdb", "--compact"]
        )
        assert result.exit_code == 0, result.output
        json_text = result.output.strip()
        assert "\n" not in json_text
        # must still be valid JSON
        json.loads(json_text)

    def test_out_flag_writes_file(self, tmp_path: Path) -> None:
        """--out writes JSON to file and exits 0."""
        out_path = tmp_path / "emit.json"
        result = runner.invoke(
            app,
            ["spec", "emit-sample", "--inputs", "test.pdb", "--out", str(out_path)],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        parsed = json.loads(out_path.read_text(encoding="utf-8"))
        assert parsed["_spec_class"] == "SamplingSpecification"

    def test_missing_inputs_exits_two(self) -> None:
        """Calling emit-sample with no --inputs exits 2."""
        result = runner.invoke(app, ["spec", "emit-sample"])
        assert result.exit_code == 2

    def test_invalid_spec_exits_one(self) -> None:
        """straight_through without iterations/learning_rate raises ValueError → exit 1."""
        result = runner.invoke(
            app,
            [
                "spec",
                "emit-sample",
                "--inputs",
                "test.pdb",
                "--sampling-strategy",
                "straight_through",
            ],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# emit-score additional tests
# ---------------------------------------------------------------------------


class TestEmitScore:
    """Tests for ``aminx spec emit-score``."""

    def test_exits_zero_produces_valid_json(self) -> None:
        """emit-score with required args exits 0 and emits valid JSON."""
        result = runner.invoke(
            app,
            [
                "spec",
                "emit-score",
                "--inputs",
                "test.pdb",
                "--sequences-to-score",
                "ACDEFGHIKLMNPQRSTVWY",
            ],
        )
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["_spec_class"] == "ScoringSpecification"

    def test_missing_sequences_exits_two(self) -> None:
        """emit-score with no --sequences-to-score exits 2."""
        result = runner.invoke(app, ["spec", "emit-score", "--inputs", "test.pdb"])
        assert result.exit_code == 2


# ---------------------------------------------------------------------------
# emit-jacobian additional tests
# ---------------------------------------------------------------------------


class TestEmitJacobian:
    """Tests for ``aminx spec emit-jacobian``."""

    def test_exits_zero_produces_valid_json(self) -> None:
        """emit-jacobian with minimal args exits 0 and emits valid JSON."""
        result = runner.invoke(app, ["spec", "emit-jacobian", "--inputs", "test.pdb"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["_spec_class"] == "JacobianSpecification"

    def test_compact_flag_produces_single_line(self) -> None:
        """--compact produces single-line JSON."""
        result = runner.invoke(
            app, ["spec", "emit-jacobian", "--inputs", "test.pdb", "--compact"]
        )
        assert result.exit_code == 0, result.output
        json_text = result.output.strip()
        assert "\n" not in json_text


# ---------------------------------------------------------------------------
# emit-inspect additional tests
# ---------------------------------------------------------------------------


class TestEmitInspect:
    """Tests for ``aminx spec emit-inspect``."""

    def test_exits_zero_produces_valid_json(self) -> None:
        """emit-inspect with minimal args exits 0 and emits valid JSON."""
        result = runner.invoke(app, ["spec", "emit-inspect", "--inputs", "test.pdb"])
        assert result.exit_code == 0, result.output
        parsed = json.loads(result.output)
        assert parsed["_spec_class"] == "InspectionSpecification"

    def test_compact_flag_produces_single_line(self) -> None:
        """--compact produces single-line JSON."""
        result = runner.invoke(
            app, ["spec", "emit-inspect", "--inputs", "test.pdb", "--compact"]
        )
        assert result.exit_code == 0, result.output
        json_text = result.output.strip()
        assert "\n" not in json_text


# ---------------------------------------------------------------------------
# spec --help lists all expected subcommands
# ---------------------------------------------------------------------------


def test_spec_help_lists_emit_subcommands() -> None:
    """``aminx spec --help`` lists all 7 subcommands including the 4 emit-* ones."""
    result = runner.invoke(app, ["spec", "--help"])
    assert result.exit_code == 0, result.output
    for cmd in (
        "validate",
        "roundtrip",
        "portable-roundtrip",
        "emit-sample",
        "emit-score",
        "emit-jacobian",
        "emit-inspect",
    ):
        assert cmd in result.output, f"Expected '{cmd}' in spec --help output"
