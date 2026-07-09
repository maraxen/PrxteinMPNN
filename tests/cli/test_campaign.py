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


def test_campaign_plan_chain_id_uniform_filter(tmp_path: Path) -> None:
    """--chain-id as a JSON array applies uniformly to every row's sampling_spec."""
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
            "--chain-id", '["A", "B"]',
        ],
    )

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}.\nOutput:\n{result.output}"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    assert len(rows) >= 1
    for row in rows:
        assert row["sampling_spec"]["chain_id"] == ["A", "B"]


def test_campaign_plan_chain_id_per_input_dict(tmp_path: Path) -> None:
    """--chain-id as a JSON object selects chains independently per --inputs path.

    Covers the multi-input (PoE) case where different structures have different
    chain layouts (e.g. an extra crystallographic copy in one input but not
    another) -- see aminx debt for the necklace-p2 multi-state chain mismatch
    this was added to resolve.
    """
    manifest_path = tmp_path / "test.manifest.json"
    output_root = tmp_path / "outputs"
    chain_id_map = {"a.pdb": ["A", "B"], "b.pdb": ["A", "C"]}

    result = runner.invoke(
        app,
        [
            "campaign",
            "plan",
            "--inputs", "a.pdb,b.pdb",
            "--campaign-id", "test",
            "--manifest-path", str(manifest_path),
            "--output-root", str(output_root),
            "--designs-per-library-type", "1",
            "--samples-chunk-size", "1",
            "--checkpoint-id", "test-checkpoint-v1",
            "--chain-id", json.dumps(chain_id_map),
        ],
    )

    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}.\nOutput:\n{result.output}"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    assert len(rows) >= 1
    for row in rows:
        assert row["sampling_spec"]["chain_id"] == chain_id_map


def test_campaign_plan_without_chain_id_omits_it(tmp_path: Path) -> None:
    """Default (no --chain-id) behavior is unchanged: chain_id stays None."""
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
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in payload["rows"]:
        assert row["sampling_spec"]["chain_id"] is None
