"""Smoke tests for the aminx campaign Typer subapp (src/aminx/cli.py).

Tests call the real public functions in campaign.py via the Typer CLI layer.
No JAX, no GPU, no model weights required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aminx.cli import app

runner = CliRunner()


def test_campaign_plan_creates_manifest(tmp_path: Path) -> None:
    """campaign plan end-to-end: writes a valid manifest with at least one row."""
    manifest_path = tmp_path / "test.manifest.json"
    output_root = tmp_path / "outputs"

    result = runner.invoke(
        app,
        [
            "campaign",
            "plan",
            "--inputs", "fake.pdb",
            "--campaign-id", "test",
            "--manifest-path", str(manifest_path),
            "--output-root", str(output_root),
            "--designs-per-library-type", "1",
            "--samples-chunk-size", "1",
            "--checkpoint-id", "test-checkpoint-v1",
        ],
    )

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}.\nOutput:\n{result.output}"
    assert manifest_path.exists(), "Manifest file was not created"

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    assert len(rows) >= 1, f"Expected at least one manifest row, got {len(rows)}"
