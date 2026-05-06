"""Tests for ``prxteinmpnn`` Typer CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from prxteinmpnn.cli import app
from prxteinmpnn.run.spec_json import run_specification_to_json
from prxteinmpnn.run.specs import RunSpecification


@pytest.fixture
def runner() -> CliRunner:
  return CliRunner()


def test_spec_validate_ok(runner: CliRunner, tmp_path: Path) -> None:
  spec = RunSpecification(inputs=["x.pdb"], tied_positions=None)
  path = tmp_path / "spec.json"
  path.write_text(run_specification_to_json(spec, indent=None), encoding="utf-8")
  result = runner.invoke(app, ["spec", "validate", str(path)])
  assert result.exit_code == 0
  assert "RunSpecification" in result.stdout


def test_spec_roundtrip_writes_file(runner: CliRunner, tmp_path: Path) -> None:
  spec = RunSpecification(inputs=["y.pdb"], tied_positions=None)
  src = tmp_path / "in.json"
  dst = tmp_path / "out.json"
  src.write_text(run_specification_to_json(spec, indent=None), encoding="utf-8")
  result = runner.invoke(app, ["spec", "roundtrip", str(src), "--out", str(dst)])
  assert result.exit_code == 0
  assert dst.is_file()
  text = dst.read_text(encoding="utf-8")
  assert "_spec_class" in text
