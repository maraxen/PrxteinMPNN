"""Tests for runtime prep checkpoint registry behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prxteinmpnn.run.prep import prep_protein_stream_and_model
from prxteinmpnn.run.specs import RunSpecification


def _write_registry(
  path: Path,
  *,
  model_family: str,
  checkpoint_id: str,
  artifact_path: str,
  sha256: str | None = None,
) -> None:
  entry: dict[str, object] = {
    "model_family": model_family,
    "checkpoint_id": checkpoint_id,
    "artifact_path": artifact_path,
  }
  if sha256 is not None:
    entry["sha256"] = sha256
  payload = {"entries": [entry]}
  path.write_text(json.dumps(payload), encoding="utf-8")


def test_prep_resolves_registry_checkpoint_for_ligand(tmp_path: Path) -> None:
  artifact_path = tmp_path / "ligand.eqx"
  artifact_path.write_bytes(b"ligand-checkpoint")
  registry_path = tmp_path / "checkpoint_registry.json"
  expected_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
  _write_registry(
    registry_path,
    model_family="ligandmpnn",
    checkpoint_id="ligandmpnn_v_32_020_25",
    artifact_path="ligand.eqx",
    sha256=expected_sha,
  )
  spec = RunSpecification(
    inputs=["dummy.pdb"],
    model_family="ligandmpnn",
    checkpoint_id="ligandmpnn_v_32_020_25",
    checkpoint_registry_path=registry_path,
  )
  fake_model = MagicMock()

  with patch("prxteinmpnn.run.prep.create_protein_dataset", return_value=[]):
    with patch("prxteinmpnn.run.prep.load_ligand_model", return_value=fake_model) as mock_load:
      with patch(
        "prxteinmpnn.run.prep.eqx.nn.inference_mode",
        side_effect=lambda model, value=True: model,
      ):
        _, model = prep_protein_stream_and_model(spec)

  assert model is fake_model
  assert mock_load.call_args.kwargs["local_path"] == str(artifact_path.resolve())


def test_prep_registry_checksum_mismatch_raises(tmp_path: Path) -> None:
  artifact_path = tmp_path / "ligand.eqx"
  artifact_path.write_bytes(b"ligand-checkpoint")
  registry_path = tmp_path / "checkpoint_registry.json"
  _write_registry(
    registry_path,
    model_family="ligandmpnn",
    checkpoint_id="ligandmpnn_v_32_020_25",
    artifact_path="ligand.eqx",
    sha256="0" * 64,
  )
  spec = RunSpecification(
    inputs=["dummy.pdb"],
    model_family="ligandmpnn",
    checkpoint_id="ligandmpnn_v_32_020_25",
    checkpoint_registry_path=registry_path,
  )

  with patch("prxteinmpnn.run.prep.create_protein_dataset", return_value=[]):
    with pytest.raises(ValueError, match="checksum mismatch"):
      prep_protein_stream_and_model(spec)


def test_prep_registry_missing_entry_raises(tmp_path: Path) -> None:
  artifact_path = tmp_path / "protein.eqx"
  artifact_path.write_bytes(b"protein-checkpoint")
  registry_path = tmp_path / "checkpoint_registry.json"
  _write_registry(
    registry_path,
    model_family="proteinmpnn",
    checkpoint_id="original_v_48_020",
    artifact_path="protein.eqx",
  )
  spec = RunSpecification(
    inputs=["dummy.pdb"],
    model_family="ligandmpnn",
    checkpoint_id="ligandmpnn_v_32_020_25",
    checkpoint_registry_path=registry_path,
  )

  with patch("prxteinmpnn.run.prep.create_protein_dataset", return_value=[]):
    with pytest.raises(ValueError, match="No checkpoint registry entry found"):
      prep_protein_stream_and_model(spec)
