"""Tests for campaign planner and worker entrypoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from prxteinmpnn.run.campaign import (
  evaluate_scale_ramp_reports,
  execute_manifest,
  evaluate_campaign_gates,
  plan_scale_ramp,
  plan_campaign_manifest,
  run_manifest_row,
  write_campaign_manifest,
)
from prxteinmpnn.run.campaign_manifest import load_manifest
from prxteinmpnn.run.specs import SamplingSpecification


def _write_mock_sampling_output(spec: SamplingSpecification) -> dict[str, str]:
  assert spec.output_h5_path is not None
  output_path = Path(spec.output_h5_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(output_path, "w") as handle:
    handle.attrs["schema_version"] = "grid_sampling_v1"
    handle.create_dataset("sample_indices", data=np.array([0], dtype=np.int32))
    group = handle.create_group("structure_0")
    group.attrs["structure_id"] = "state_0"
    group.create_dataset("sequences", data=np.zeros((1, 1, 1, 4), dtype=np.int32))
  return {"status": "ok", "output_h5_path": str(output_path)}


def _first_manifest_row(manifest_path: Path) -> dict[str, object]:
  payload = load_manifest(manifest_path)
  rows = payload["rows"]
  assert isinstance(rows, list)
  first_row = rows[0]
  assert isinstance(first_row, dict)
  return first_row


def _run_all_rows_with_mock_writer(manifest_path: Path) -> None:
  payload = load_manifest(manifest_path)
  rows = payload["rows"]
  assert isinstance(rows, list)
  with patch("prxteinmpnn.run.campaign.sample", side_effect=_write_mock_sampling_output):
    for row_index in range(len(rows)):
      result = run_manifest_row(manifest_path, row_index=row_index)
      assert result["status"] == "ok"


def test_plan_campaign_manifest_generates_full_library_matrix() -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  rows = plan_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    designs_per_library_type=5,
    samples_chunk_size=2,
    output_root="outputs",
  )

  assert len(rows) == 24  # 8 library types * ceil(5/2)=3 chunks
  assert len({row["manifest_row_hash"] for row in rows}) == len(rows)
  assert all(row["schema_version"] == "campaign_manifest_v1" for row in rows)
  assert all("sampling_spec" in row for row in rows)


def test_run_manifest_row_executes_selected_row(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs",
  )

  row = _first_manifest_row(manifest_path)
  output_path = Path(str(row["output_h5_path"]))
  done_marker_path = output_path.with_name(f"{output_path.name}.done.json")

  with patch("prxteinmpnn.run.campaign.sample", side_effect=_write_mock_sampling_output) as mock_sample:
    result = run_manifest_row(manifest_path, row_index=0)

  assert result["status"] == "ok"
  assert result["output_h5_path"] == str(output_path.resolve())
  assert output_path.exists()
  assert done_marker_path.exists()
  done_payload = json.loads(done_marker_path.read_text(encoding="utf-8"))
  assert done_payload["manifest_row_hash"] == row["manifest_row_hash"]
  assert not output_path.with_name(f"{output_path.name}.lock").exists()
  assert not list(output_path.parent.glob(f"{output_path.name}.partial.*"))
  called_spec = mock_sample.call_args.args[0]
  assert isinstance(called_spec, SamplingSpecification)
  assert called_spec.grid_mode is True
  assert called_spec.campaign_mode is True
  assert called_spec.return_logits is False
  assert called_spec.output_h5_path != output_path.resolve()


def test_run_manifest_row_is_idempotent_with_done_marker(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs",
  )

  with patch("prxteinmpnn.run.campaign.sample", side_effect=_write_mock_sampling_output):
    first = run_manifest_row(manifest_path, row_index=0)
  assert first["status"] == "ok"

  with patch("prxteinmpnn.run.campaign.sample") as mock_sample:
    second = run_manifest_row(manifest_path, row_index=0)
  mock_sample.assert_not_called()
  assert second["status"] == "already_done"


def test_run_manifest_row_fails_when_active_lock_exists(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs",
  )
  row = _first_manifest_row(manifest_path)
  output_path = Path(str(row["output_h5_path"]))
  lock_path = output_path.with_name(f"{output_path.name}.lock")
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  now = time.time()
  lock_path.write_text(
    json.dumps(
      {
        "schema_version": "campaign_lock_v1",
        "owner_token": "external-owner",
        "manifest_row_hash": row["manifest_row_hash"],
        "attempt_id": "external-attempt",
        "created_at_unix_s": now,
        "heartbeat_at_unix_s": now,
        "lease_expires_at_unix_s": now + 600.0,
      },
      sort_keys=True,
      separators=(",", ":"),
    ),
    encoding="utf-8",
  )

  with patch("prxteinmpnn.run.campaign.sample", side_effect=_write_mock_sampling_output):
    with pytest.raises(RuntimeError, match="Active lock already held"):
      run_manifest_row(manifest_path, row_index=0)


def test_run_manifest_row_steals_expired_lock(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs",
  )
  row = _first_manifest_row(manifest_path)
  output_path = Path(str(row["output_h5_path"]))
  lock_path = output_path.with_name(f"{output_path.name}.lock")
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  now = time.time()
  lock_path.write_text(
    json.dumps(
      {
        "schema_version": "campaign_lock_v1",
        "owner_token": "expired-owner",
        "manifest_row_hash": row["manifest_row_hash"],
        "attempt_id": "expired-attempt",
        "created_at_unix_s": now - 1200.0,
        "heartbeat_at_unix_s": now - 1200.0,
        "lease_expires_at_unix_s": now - 10.0,
      },
      sort_keys=True,
      separators=(",", ":"),
    ),
    encoding="utf-8",
  )

  with patch("prxteinmpnn.run.campaign.sample", side_effect=_write_mock_sampling_output):
    result = run_manifest_row(manifest_path, row_index=0)
  assert result["status"] == "ok"
  assert not lock_path.exists()


def test_run_manifest_row_requires_backend_for_distributed_lock(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs",
  )

  with pytest.raises(ValueError, match="requires a DistributedLockBackend implementation"):
    run_manifest_row(manifest_path, row_index=0, lock_backend="distributed")


def test_evaluate_campaign_gates_requires_rerun_by_default(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs",
  )
  _run_all_rows_with_mock_writer(manifest_path)

  report = evaluate_campaign_gates(manifest_path)
  assert report["gates"]["zero_crashes"] is True
  assert report["gates"]["metadata_complete"] is True
  assert report["gates"]["determinism_pass"] is False
  assert report["promote"] is False


def test_evaluate_campaign_gates_passes_with_matching_rerun(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs_main",
  )
  rerun_manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest_rerun.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs_rerun",
  )
  _run_all_rows_with_mock_writer(manifest_path)
  _run_all_rows_with_mock_writer(rerun_manifest_path)

  report = evaluate_campaign_gates(
    manifest_path,
    rerun_manifest_paths=(rerun_manifest_path,),
  )
  assert report["gates"]["determinism_pass"] is True
  assert report["reruns"]["full_cell_rerun"] is True
  assert report["promote"] is True


def test_execute_manifest_runs_all_rows(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs",
  )
  payload = load_manifest(manifest_path)
  rows = payload["rows"]
  assert isinstance(rows, list)

  with patch("prxteinmpnn.run.campaign.sample", side_effect=_write_mock_sampling_output):
    summary = execute_manifest(manifest_path)

  assert summary["selected_rows"] == len(rows)
  assert summary["successful_rows"] == len(rows)
  assert summary["failed_rows"] == 0


def test_execute_manifest_continue_on_error_collects_failures(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  manifest_path = write_campaign_manifest(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_path=tmp_path / "campaign_manifest.json",
    designs_per_library_type=3,
    samples_chunk_size=2,
    output_root=tmp_path / "outputs",
  )

  def _fail_all(_: SamplingSpecification) -> dict[str, str]:
    msg = "boom"
    raise RuntimeError(msg)

  with patch("prxteinmpnn.run.campaign.sample", side_effect=_fail_all):
    summary = execute_manifest(manifest_path, continue_on_error=True)

  assert summary["failed_rows"] == summary["selected_rows"]
  assert summary["successful_rows"] == 0
  assert all(row["status"] == "error" for row in summary["row_results"])


def test_plan_scale_ramp_writes_stage_manifests(tmp_path: Path) -> None:
  base_spec = SamplingSpecification(
    inputs=["tests/data/1ubq.pdb"],
    return_logits=False,
    model_family="proteinmpnn",
  )
  plan_payload = plan_scale_ramp(
    base_spec=base_spec,
    campaign_id="tev-campaign",
    manifest_dir=tmp_path / "manifests",
    output_root=tmp_path / "outputs",
    stage_designs_per_library_type=(3, 5),
    samples_chunk_size=2,
  )

  assert plan_payload["schema_version"] == "campaign_scale_ramp_plan_v1"
  assert len(plan_payload["stages"]) == 2
  first_manifest = Path(plan_payload["stages"][0]["manifest_path"])
  first_payload = load_manifest(first_manifest)
  metadata = first_payload.get("metadata")
  assert isinstance(metadata, dict)
  assert metadata["designs_per_library_type"] == 3


def test_evaluate_scale_ramp_reports_detects_rollback(tmp_path: Path) -> None:
  stage0 = {
    "schema_version": "campaign_gate_report_v1",
    "manifest_path": "stage0.json",
    "promote": True,
    "gates": {
      "zero_crashes": True,
      "zero_lineage_collisions": True,
      "metadata_complete": True,
      "checkpoint_preflight_pass": True,
      "determinism_pass": True,
      "resource_envelope_within_limits": True,
    },
    "base": {"runtime_issues": []},
  }
  stage1 = {
    "schema_version": "campaign_gate_report_v1",
    "manifest_path": "stage1.json",
    "promote": False,
    "gates": {
      "zero_crashes": True,
      "zero_lineage_collisions": True,
      "metadata_complete": True,
      "checkpoint_preflight_pass": True,
      "determinism_pass": False,
      "resource_envelope_within_limits": True,
    },
    "base": {"runtime_issues": []},
  }
  stage0_path = tmp_path / "stage0.report.json"
  stage1_path = tmp_path / "stage1.report.json"
  stage0_path.write_text(json.dumps(stage0), encoding="utf-8")
  stage1_path.write_text(json.dumps(stage1), encoding="utf-8")

  summary = evaluate_scale_ramp_reports((stage0_path, stage1_path))
  assert summary["schema_version"] == "campaign_scale_ramp_report_v1"
  assert summary["promote"] is False
  assert summary["rollback_stage_index"] == 1
  assert "determinism_drift" in summary["rollback_reasons"]
