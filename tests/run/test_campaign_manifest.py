"""Tests for campaign manifest schema and validation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from prxteinmpnn.run.campaign_manifest import (
  CAMPAIGN_MANIFEST_SCHEMA_VERSION,
  build_manifest_row,
  load_manifest,
  manifest_row_hash,
  validate_manifest_rows,
  write_manifest,
)
from prxteinmpnn.run.specs import SamplingSpecification


def _base_spec() -> SamplingSpecification:
  return SamplingSpecification(
    inputs=["dummy.pdb"],
    model_family="ligandmpnn",
    checkpoint_id="ligandmpnn_v_32_020_25",
    temperature=(0.1, 0.3),
    backbone_noise=(0.0, 0.2),
    return_logits=False,
  )


def test_build_manifest_row_adds_hash_and_canonical_fields() -> None:
  spec = _base_spec()
  row = build_manifest_row(
    spec,
    campaign_id="tev-campaign",
    job_index=4,
    chunk_id=2,
    sample_start=100,
    sample_count=50,
    fixed_policy="catalytic_triad",
    state_weight_profile="equal",
    planner_version="planner-v1",
    dataset_fingerprint="dataset-fp",
    environment_image="ghcr.io/example/image:1",
    git_sha="abc123",
    config_hash="cfg123",
  )
  assert row["schema_version"] == CAMPAIGN_MANIFEST_SCHEMA_VERSION
  assert row["temperature"] == ["0.10000000000000001", "0.29999999999999999"]
  assert row["backbone_noise"] == ["0", "0.20000000000000001"]
  assert row["manifest_row_hash"] == manifest_row_hash(row)


def test_validate_manifest_rows_rejects_duplicate_hashes() -> None:
  spec = _base_spec()
  row = build_manifest_row(
    spec,
    campaign_id="tev-campaign",
    job_index=0,
    chunk_id=0,
    sample_start=0,
    sample_count=10,
    fixed_policy="catalytic_triad",
    state_weight_profile="equal",
  )
  with pytest.raises(ValueError, match="Duplicate manifest_row_hash"):
    validate_manifest_rows([row, dict(row)])


def test_validate_manifest_rows_requires_full_matrix() -> None:
  spec = _base_spec()
  partial_rows = [
    build_manifest_row(
      spec,
      campaign_id="tev-campaign",
      job_index=idx,
      chunk_id=idx,
      sample_start=idx * 10,
      sample_count=10,
      fixed_policy="catalytic_triad",
      state_weight_profile="equal",
      job_id=f"job-{idx}",
    )
    for idx in range(4)
  ]
  with pytest.raises(ValueError, match="missing library matrix combinations"):
    validate_manifest_rows(
      partial_rows,
      required_fixed_policies=("catalytic_triad", "active_site"),
    )


def test_write_and_load_manifest_roundtrip(tmp_path: Path) -> None:
  spec = _base_spec()
  rows = []
  index = 0
  for fixed_policy in ("catalytic_triad", "active_site"):
    for ligand_on in (False, True):
      for sidechain_on in (False, True):
        spec_variant = SamplingSpecification(
          inputs=["dummy.pdb"],
          model_family="ligandmpnn",
          checkpoint_id="ligandmpnn_v_32_020_25",
          ligand_conditioning=ligand_on,
          sidechain_conditioning=sidechain_on,
          return_logits=False,
        )
        rows.append(
          build_manifest_row(
            spec_variant,
            campaign_id="tev-campaign",
            job_index=index,
            chunk_id=index,
            sample_start=index * 10,
            sample_count=10,
            fixed_policy=fixed_policy,
            state_weight_profile="equal",
            job_id=f"job-{index}",
          ),
        )
        index += 1

  validate_manifest_rows(rows, required_fixed_policies=("catalytic_triad", "active_site"))
  output_path = write_manifest(tmp_path / "manifest.json", rows, metadata={"owner": "tests"})
  payload = load_manifest(output_path)

  assert payload["schema_version"] == CAMPAIGN_MANIFEST_SCHEMA_VERSION
  assert len(payload["rows"]) == 8
  assert payload["metadata"]["owner"] == "tests"
