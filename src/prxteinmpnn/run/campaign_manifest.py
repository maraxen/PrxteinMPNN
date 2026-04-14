"""Campaign manifest schema and validation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from prxteinmpnn.io.weights import LIGAND_DEFAULT_CHECKPOINT
from prxteinmpnn.run.specs import SamplingSpecification

CAMPAIGN_MANIFEST_SCHEMA_VERSION = "campaign_manifest_v1"

REQUIRED_ROW_KEYS = {
  "schema_version",
  "campaign_id",
  "job_id",
  "job_index",
  "chunk_id",
  "sample_start",
  "sample_count",
  "model_family",
  "checkpoint_id",
  "ligand_conditioning",
  "sidechain_conditioning",
  "fixed_policy",
  "state_weight_profile",
  "multi_state_strategy",
  "temperature",
  "backbone_noise",
  "planner_version",
  "dataset_fingerprint",
  "environment_image",
  "git_sha",
  "config_hash",
  "manifest_row_hash",
}
HASH_ROW_KEYS = tuple(sorted(REQUIRED_ROW_KEYS - {"manifest_row_hash"}))


def _canonical_float_sequence(values: Any) -> list[str]:  # noqa: ANN401
  return [format(float(value), ".17g") for value in values]


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  ).encode("utf-8")


def manifest_row_hash(row: dict[str, Any]) -> str:
  """Return deterministic SHA256 hash for a manifest row payload."""
  payload = {key: row[key] for key in HASH_ROW_KEYS}
  return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _default_checkpoint_id(spec: SamplingSpecification) -> str:
  if spec.model_family == "ligandmpnn":
    return spec.checkpoint_id or LIGAND_DEFAULT_CHECKPOINT
  return f"{spec.model_weights}_{spec.model_version}"


def build_manifest_row(
  spec: SamplingSpecification,
  *,
  campaign_id: str,
  job_index: int,
  chunk_id: int,
  sample_start: int,
  sample_count: int,
  fixed_policy: str,
  state_weight_profile: str,
  planner_version: str = "manual",
  dataset_fingerprint: str = "unknown",
  environment_image: str = "unknown",
  git_sha: str = "unknown",
  config_hash: str = "unknown",
  checkpoint_id: str | None = None,
  job_id: str | None = None,
) -> dict[str, Any]:
  """Build a canonical campaign manifest row and attach deterministic hash."""
  resolved_checkpoint = checkpoint_id or _default_checkpoint_id(spec)
  resolved_job_id = job_id or spec.job_id or f"{campaign_id}-job-{job_index}"
  row: dict[str, Any] = {
    "schema_version": CAMPAIGN_MANIFEST_SCHEMA_VERSION,
    "campaign_id": campaign_id,
    "job_id": resolved_job_id,
    "job_index": int(job_index),
    "chunk_id": int(chunk_id),
    "sample_start": int(sample_start),
    "sample_count": int(sample_count),
    "model_family": spec.model_family,
    "checkpoint_id": resolved_checkpoint,
    "checkpoint_registry_path": (
      str(spec.checkpoint_registry_path) if spec.checkpoint_registry_path is not None else None
    ),
    "model_local_path": str(spec.model_local_path) if spec.model_local_path is not None else None,
    "ligand_conditioning": bool(spec.ligand_conditioning),
    "sidechain_conditioning": bool(spec.sidechain_conditioning),
    "fixed_policy": fixed_policy,
    "state_weight_profile": state_weight_profile,
    "multi_state_strategy": spec.multi_state_strategy,
    "temperature": _canonical_float_sequence(spec.temperature),
    "backbone_noise": _canonical_float_sequence(spec.backbone_noise),
    "planner_version": planner_version,
    "dataset_fingerprint": dataset_fingerprint,
    "environment_image": environment_image,
    "git_sha": git_sha,
    "config_hash": config_hash,
  }
  row["manifest_row_hash"] = manifest_row_hash(row)
  return row


def validate_manifest_rows(
  rows: list[dict[str, Any]],
  *,
  required_fixed_policies: tuple[str, ...] | None = None,
) -> None:
  """Validate campaign manifest rows and lineage/hash integrity."""
  if not rows:
    msg = "Campaign manifest must contain at least one row."
    raise ValueError(msg)

  seen_hashes: set[str] = set()
  seen_lineages: set[tuple[str, str, int, int, int]] = set()
  observed_matrix: set[tuple[bool, bool, str]] = set()

  for row in rows:
    missing_keys = REQUIRED_ROW_KEYS - row.keys()
    if missing_keys:
      msg = f"Manifest row is missing required keys: {sorted(missing_keys)}"
      raise ValueError(msg)
    if row["schema_version"] != CAMPAIGN_MANIFEST_SCHEMA_VERSION:
      msg = (
        "Manifest row schema mismatch: "
        f"expected {CAMPAIGN_MANIFEST_SCHEMA_VERSION!r}, got {row['schema_version']!r}."
      )
      raise ValueError(msg)

    observed_hash = str(row["manifest_row_hash"])
    expected_hash = manifest_row_hash(row)
    if observed_hash != expected_hash:
      msg = (
        "Manifest row hash mismatch for "
        f"job_id={row['job_id']!r}, chunk_id={row['chunk_id']}: "
        f"expected {expected_hash}, observed {observed_hash}."
      )
      raise ValueError(msg)
    if observed_hash in seen_hashes:
      msg = f"Duplicate manifest_row_hash detected: {observed_hash}"
      raise ValueError(msg)
    seen_hashes.add(observed_hash)

    lineage_key = (
      str(row["campaign_id"]),
      str(row["job_id"]),
      int(row["chunk_id"]),
      int(row["sample_start"]),
      int(row["sample_count"]),
    )
    if lineage_key in seen_lineages:
      msg = (
        "Duplicate lineage tuple detected: "
        f"campaign_id={lineage_key[0]!r}, job_id={lineage_key[1]!r}, "
        f"chunk_id={lineage_key[2]}, sample_start={lineage_key[3]}, sample_count={lineage_key[4]}."
      )
      raise ValueError(msg)
    seen_lineages.add(lineage_key)

    observed_matrix.add(
      (
        bool(row["ligand_conditioning"]),
        bool(row["sidechain_conditioning"]),
        str(row["fixed_policy"]),
      ),
    )

  if required_fixed_policies is not None:
    expected_matrix = {
      (ligand_on, sidechain_on, fixed_policy)
      for fixed_policy in required_fixed_policies
      for ligand_on in (False, True)
      for sidechain_on in (False, True)
    }
    missing = expected_matrix - observed_matrix
    if missing:
      msg = f"Manifest is missing library matrix combinations: {sorted(missing)}"
      raise ValueError(msg)


def write_manifest(
  path: str | Path,
  rows: list[dict[str, Any]],
  *,
  metadata: dict[str, Any] | None = None,
) -> Path:
  """Validate and write campaign manifest to disk."""
  validate_manifest_rows(rows)
  payload: dict[str, Any] = {
    "schema_version": CAMPAIGN_MANIFEST_SCHEMA_VERSION,
    "rows": rows,
  }
  if metadata is not None:
    payload["metadata"] = metadata

  output_path = Path(path)
  output_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
  return output_path


def load_manifest(path: str | Path) -> dict[str, Any]:
  """Load manifest JSON payload from disk."""
  payload = json.loads(Path(path).read_text(encoding="utf-8"))
  if payload.get("schema_version") != CAMPAIGN_MANIFEST_SCHEMA_VERSION:
    msg = (
      "Manifest schema mismatch: "
      f"expected {CAMPAIGN_MANIFEST_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}."
    )
    raise ValueError(msg)
  rows = payload.get("rows")
  if not isinstance(rows, list):
    msg = "Manifest payload must contain a 'rows' list."
    raise ValueError(msg)
  validate_manifest_rows(rows)
  return payload
