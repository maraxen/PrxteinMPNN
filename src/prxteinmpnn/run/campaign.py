"""Planner and worker entrypoints for scheduler-agnostic campaign manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
import unicodedata
import uuid
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np

from prxteinmpnn.run.campaign_manifest import (
  build_manifest_row,
  load_manifest,
  validate_manifest_rows,
  write_manifest,
)
from prxteinmpnn.run.sampling import sample
from prxteinmpnn.run.specs import SamplingSpecification

logger = logging.getLogger(__name__)

LOCK_SCHEMA_VERSION = "campaign_lock_v1"
DONE_MARKER_SCHEMA_VERSION = "campaign_done_marker_v1"
DEFAULT_LOCK_LEASE_SECONDS = 1800
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60
GateCellKey = tuple[str, str, str, tuple[str, ...], tuple[str, ...], bool, bool]
ScaleStageResult = dict[str, Any]


@dataclass
class ManifestGateState:
  """Gate-relevant status collected from a campaign manifest."""

  manifest_path: Path
  total_rows: int
  completed_rows: int
  digest_by_hash: dict[str, str]
  cell_to_hashes: dict[GateCellKey, set[str]]
  metadata_issues: list[str]
  checkpoint_issues: list[str]
  runtime_issues: list[str]


class DistributedLockBackend(Protocol):
  """Distributed lock backend interface used by campaign workers."""

  def acquire(self, *, lock_key: str, owner_token: str, lease_seconds: int) -> None:
    """Acquire a distributed lock key for an owner token."""

  def heartbeat(self, *, lock_key: str, owner_token: str, lease_seconds: int) -> None:
    """Refresh lease ownership for an active lock."""

  def release(self, *, lock_key: str, owner_token: str) -> None:
    """Release an owned lock key."""


def _normalize_inputs(inputs: Any) -> list[str]:  # noqa: ANN401
  if isinstance(inputs, str):
    return [inputs]
  try:
    normalized = [str(item) for item in inputs]
  except TypeError as exc:  # pragma: no cover - defensive
    msg = "Campaign planner requires string-like inputs."
    raise ValueError(msg) from exc
  if not normalized:
    msg = "Campaign planner requires at least one input structure path."
    raise ValueError(msg)
  return normalized


def _row_output_path(base_dir: Path, campaign_id: str, row_hash: str) -> str:
  return str((base_dir / campaign_id / f"{row_hash}.h5").resolve())


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
  return json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  ).encode("utf-8")


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _normalize_json_value(value: Any) -> Any:  # noqa: ANN401
  normalized: Any = value
  if isinstance(value, np.generic):
    normalized = _normalize_json_value(value.item())
  elif isinstance(value, np.ndarray):
    normalized = [_normalize_json_value(item) for item in value.tolist()]
  elif isinstance(value, (list, tuple)):
    normalized = [_normalize_json_value(item) for item in value]
  elif isinstance(value, dict):
    normalized = {str(key): _normalize_json_value(item) for key, item in sorted(value.items())}
  elif isinstance(value, bytes):
    normalized = value.decode("utf-8")
  elif isinstance(value, str):
    normalized = unicodedata.normalize("NFC", value)
  return normalized


def _update_array_digest(digest: hashlib._Hash, array: np.ndarray) -> None:
  digest.update(array.dtype.name.encode("utf-8"))
  digest.update(b"|")
  digest.update(str(array.shape).encode("utf-8"))
  digest.update(b"|")
  canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
  digest.update(canonical.tobytes(order="C"))


def _update_h5_node_digest(digest: hashlib._Hash, node: h5py.Group | h5py.Dataset, path: str) -> None:
  digest.update(path.encode("utf-8"))
  digest.update(b"\n")
  attrs_payload = {
    str(key): _normalize_json_value(node.attrs[key])  # type: ignore[index]
    for key in sorted(node.attrs.keys())
  }
  digest.update(_canonical_json_bytes(attrs_payload))
  digest.update(b"\n")
  if isinstance(node, h5py.Dataset):
    _update_array_digest(digest, np.asarray(node))
    digest.update(b"\n")
    return
  for key in sorted(node.keys()):
    child = node[key]
    _update_h5_node_digest(digest, child, f"{path}/{key}")


def _h5_content_digest(path: Path) -> str:
  digest = hashlib.sha256()
  with h5py.File(path, "r") as handle:
    _update_h5_node_digest(digest, handle, "/")
  return digest.hexdigest()


def _fsync_file(path: Path) -> None:
  with path.open("rb") as handle:
    os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
  fd = os.open(path, os.O_RDONLY)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


def _lock_path(output_h5_path: Path) -> Path:
  return output_h5_path.with_name(f"{output_h5_path.name}.lock")


def _done_marker_path(output_h5_path: Path) -> Path:
  return output_h5_path.with_name(f"{output_h5_path.name}.done.json")


def _partial_output_path(output_h5_path: Path, attempt_id: str) -> Path:
  return output_h5_path.with_name(f"{output_h5_path.name}.partial.{attempt_id}")


def _write_lock_file_exclusive(path: Path, payload: dict[str, Any]) -> None:
  lock_bytes = _canonical_json_bytes(payload)
  fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
  try:
    os.write(fd, lock_bytes)
    os.fsync(fd)
  finally:
    os.close(fd)
  _fsync_directory(path.parent)


def _write_lock_file_atomic(path: Path, payload: dict[str, Any]) -> None:
  tmp_path = path.with_name(f"{path.name}.tmp.{uuid.uuid4().hex}")
  tmp_path.write_bytes(_canonical_json_bytes(payload))
  _fsync_file(tmp_path)
  tmp_path.replace(path)
  _fsync_directory(path.parent)


def _read_lock_file(path: Path) -> tuple[dict[str, Any], bytes]:
  raw = path.read_bytes()
  parsed = json.loads(raw.decode("utf-8"))
  if not isinstance(parsed, dict):
    msg = f"Invalid lock payload at {path}: expected object."
    raise TypeError(msg)
  return parsed, raw


def _acquire_local_fs_lock(
  *,
  lock_path: Path,
  owner_token: str,
  manifest_row_hash: str,
  attempt_id: str,
  lease_seconds: int,
) -> None:
  now = time.time()
  payload: dict[str, Any] = {
    "schema_version": LOCK_SCHEMA_VERSION,
    "owner_token": owner_token,
    "manifest_row_hash": manifest_row_hash,
    "attempt_id": attempt_id,
    "created_at_unix_s": now,
    "heartbeat_at_unix_s": now,
    "lease_expires_at_unix_s": now + lease_seconds,
  }
  try:
    _write_lock_file_exclusive(lock_path, payload)
  except FileExistsError as exc:
    existing, existing_raw = _read_lock_file(lock_path)
    expires_at = float(existing.get("lease_expires_at_unix_s", 0.0))
    if expires_at >= now:
      msg = (
        f"Active lock already held for output {lock_path}: "
        f"owner={existing.get('owner_token')!r}, expires_at={expires_at}."
      )
      raise RuntimeError(msg) from exc

    # Stale-lock recovery with compare-and-swap precondition:
    # only steal if the lock bytes are unchanged from the observed stale state.
    tmp_payload_path = lock_path.with_name(f"{lock_path.name}.steal.{attempt_id}.tmp")
    tmp_payload_path.write_bytes(_canonical_json_bytes(payload))
    _fsync_file(tmp_payload_path)
    current_raw = lock_path.read_bytes()
    if current_raw != existing_raw:
      tmp_payload_path.unlink(missing_ok=True)
      msg = f"Lock at {lock_path} changed during stale-lock recovery; aborting steal."
      raise RuntimeError(msg) from exc
    tmp_payload_path.replace(lock_path)
    _fsync_directory(lock_path.parent)
    observed, _ = _read_lock_file(lock_path)
    if observed.get("owner_token") != owner_token:
      msg = f"Failed to claim stale lock at {lock_path}: ownership did not transfer."
      raise RuntimeError(msg) from exc


def _heartbeat_local_fs_lock(
  *,
  lock_path: Path,
  owner_token: str,
  lease_seconds: int,
) -> None:
  lock_payload, _ = _read_lock_file(lock_path)
  if lock_payload.get("owner_token") != owner_token:
    msg = (
      f"Cannot refresh lock {lock_path}: owner mismatch "
      f"(expected {owner_token!r}, observed {lock_payload.get('owner_token')!r})."
    )
    raise RuntimeError(msg)
  now = time.time()
  lock_payload["heartbeat_at_unix_s"] = now
  lock_payload["lease_expires_at_unix_s"] = now + lease_seconds
  _write_lock_file_atomic(lock_path, lock_payload)


def _release_local_fs_lock(*, lock_path: Path, owner_token: str) -> None:
  if not lock_path.exists():
    return
  lock_payload, _ = _read_lock_file(lock_path)
  if lock_payload.get("owner_token") != owner_token:
    msg = (
      f"Cannot release lock {lock_path}: owner mismatch "
      f"(expected {owner_token!r}, observed {lock_payload.get('owner_token')!r})."
    )
    raise RuntimeError(msg)
  lock_path.unlink()
  _fsync_directory(lock_path.parent)


@contextmanager
def _campaign_lock_context(  # noqa: PLR0915
  *,
  lock_backend: str,
  distributed_lock_backend: DistributedLockBackend | None,
  lock_key_path: Path,
  owner_token: str,
  manifest_row_hash: str,
  attempt_id: str,
  lease_seconds: int,
  heartbeat_interval_seconds: int,
) -> Generator[list[Exception], None, None]:
  heartbeat_errors: list[Exception] = []
  stop_event = threading.Event()

  if lock_backend == "local_fs":
    _acquire_local_fs_lock(
      lock_path=lock_key_path,
      owner_token=owner_token,
      manifest_row_hash=manifest_row_hash,
      attempt_id=attempt_id,
      lease_seconds=lease_seconds,
    )

    def _heartbeat() -> None:
      while not stop_event.wait(heartbeat_interval_seconds):
        try:
          _heartbeat_local_fs_lock(
            lock_path=lock_key_path,
            owner_token=owner_token,
            lease_seconds=lease_seconds,
          )
        except Exception as exc:  # noqa: BLE001
          heartbeat_errors.append(exc)
          return

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
      yield heartbeat_errors
    finally:
      stop_event.set()
      heartbeat_thread.join(timeout=heartbeat_interval_seconds + 1)
      try:
        _release_local_fs_lock(lock_path=lock_key_path, owner_token=owner_token)
      except Exception:
        if not heartbeat_errors:
          raise
        logger.exception("Failed to release local lock after heartbeat failure.")
    return

  if lock_backend != "distributed":
    msg = f"Unknown lock backend: {lock_backend!r}."
    raise ValueError(msg)
  if distributed_lock_backend is None:
    msg = (
      "lock_backend='distributed' requires a DistributedLockBackend implementation "
      "when calling run_manifest_row from Python."
    )
    raise ValueError(msg)
  lock_key = str(lock_key_path.resolve())
  distributed_lock_backend.acquire(
    lock_key=lock_key,
    owner_token=owner_token,
    lease_seconds=lease_seconds,
  )

  def _distributed_heartbeat() -> None:
    while not stop_event.wait(heartbeat_interval_seconds):
      try:
        distributed_lock_backend.heartbeat(
          lock_key=lock_key,
          owner_token=owner_token,
          lease_seconds=lease_seconds,
        )
      except Exception as exc:  # noqa: BLE001
        heartbeat_errors.append(exc)
        return

  heartbeat_thread = threading.Thread(target=_distributed_heartbeat, daemon=True)
  heartbeat_thread.start()
  try:
    yield heartbeat_errors
  finally:
    stop_event.set()
    heartbeat_thread.join(timeout=heartbeat_interval_seconds + 1)
    try:
      distributed_lock_backend.release(lock_key=lock_key, owner_token=owner_token)
    except Exception:
      if not heartbeat_errors:
        raise
      logger.exception("Failed to release distributed lock after heartbeat failure.")


def _read_done_marker(path: Path) -> dict[str, Any] | None:
  if not path.exists():
    return None
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    msg = f"Done marker at {path} must be a JSON object."
    raise TypeError(msg)
  return payload


def _validate_done_marker(
  *,
  marker: dict[str, Any],
  marker_path: Path,
  output_h5_path: Path,
  manifest_row_hash: str,
) -> None:
  if marker.get("schema_version") != DONE_MARKER_SCHEMA_VERSION:
    msg = (
      f"Done marker schema mismatch at {marker_path}: "
      f"expected {DONE_MARKER_SCHEMA_VERSION!r}, got {marker.get('schema_version')!r}."
    )
    raise ValueError(msg)
  if marker.get("manifest_row_hash") != manifest_row_hash:
    msg = (
      f"Done marker manifest hash mismatch at {marker_path}: "
      f"expected {manifest_row_hash!r}, got {marker.get('manifest_row_hash')!r}."
    )
    raise ValueError(msg)
  if not output_h5_path.exists():
    msg = (
      f"Done marker exists at {marker_path} but output file is missing: {output_h5_path}."
    )
    raise ValueError(msg)
  observed_file_hash = _sha256_file(output_h5_path)
  expected_file_hash = marker.get("artifact_sha256")
  if observed_file_hash != expected_file_hash:
    msg = (
      f"Done marker artifact hash mismatch at {marker_path}: "
      f"expected {expected_file_hash!r}, observed {observed_file_hash!r}."
    )
    raise ValueError(msg)
  observed_content_digest = _h5_content_digest(output_h5_path)
  expected_content_digest = marker.get("content_digest_sha256")
  if observed_content_digest != expected_content_digest:
    msg = (
      f"Done marker content digest mismatch at {marker_path}: "
      f"expected {expected_content_digest!r}, observed {observed_content_digest!r}."
    )
    raise ValueError(msg)


def _write_done_marker(
  *,
  marker_path: Path,
  output_h5_path: Path,
  manifest_row_hash: str,
  attempt_id: str,
  artifact_sha256: str,
  content_digest_sha256: str,
  lock_backend: str,
) -> None:
  marker_payload = {
    "schema_version": DONE_MARKER_SCHEMA_VERSION,
    "manifest_row_hash": manifest_row_hash,
    "attempt_id": attempt_id,
    "output_h5_path": str(output_h5_path.resolve()),
    "artifact_sha256": artifact_sha256,
    "content_digest_sha256": content_digest_sha256,
    "lock_backend": lock_backend,
    "completed_at_unix_s": time.time(),
  }
  tmp_marker_path = marker_path.with_name(f"{marker_path.name}.tmp.{attempt_id}")
  tmp_marker_path.write_bytes(_canonical_json_bytes(marker_payload))
  _fsync_file(tmp_marker_path)
  tmp_marker_path.replace(marker_path)
  _fsync_directory(marker_path.parent)


def plan_campaign_manifest(
  *,
  base_spec: SamplingSpecification,
  campaign_id: str,
  designs_per_library_type: int,
  samples_chunk_size: int,
  output_root: str | Path,
  fixed_policies: tuple[str, ...] = ("catalytic_triad", "active_site"),
  state_weight_profiles: tuple[str, ...] = ("equal",),
  planner_version: str = "planner_v1",
  dataset_fingerprint: str = "unknown",
  environment_image: str = "unknown",
  git_sha: str = "unknown",
  config_hash: str = "unknown",
) -> list[dict[str, Any]]:
  """Plan campaign rows for all library/fixed-policy/profile combinations."""
  if designs_per_library_type <= 0:
    msg = "designs_per_library_type must be positive."
    raise ValueError(msg)
  if samples_chunk_size <= 0:
    msg = "samples_chunk_size must be positive."
    raise ValueError(msg)

  output_root_path = Path(output_root)
  rows: list[dict[str, Any]] = []
  job_index = 0

  for fixed_policy in fixed_policies:
    for state_weight_profile in state_weight_profiles:
      for ligand_on in (False, True):
        for sidechain_on in (False, True):
          spec_variant = replace(
            base_spec,
            inputs=_normalize_inputs(base_spec.inputs),
            campaign_mode=True,
            return_logits=False,
            ligand_conditioning=ligand_on,
            sidechain_conditioning=sidechain_on,
          )
          for chunk_index, sample_start in enumerate(
            range(0, designs_per_library_type, samples_chunk_size),
          ):
            sample_count = min(samples_chunk_size, designs_per_library_type - sample_start)
            row = build_manifest_row(
              spec_variant,
              campaign_id=campaign_id,
              job_index=job_index,
              chunk_id=chunk_index,
              sample_start=sample_start,
              sample_count=sample_count,
              fixed_policy=fixed_policy,
              state_weight_profile=state_weight_profile,
              planner_version=planner_version,
              dataset_fingerprint=dataset_fingerprint,
              environment_image=environment_image,
              git_sha=git_sha,
              config_hash=config_hash,
              job_id=f"{campaign_id}-job-{job_index}",
            )
            row["output_h5_path"] = _row_output_path(
              output_root_path,
              campaign_id=campaign_id,
              row_hash=row["manifest_row_hash"],
            )
            row["sampling_spec"] = {
              "inputs": _normalize_inputs(base_spec.inputs),
              "model_family": spec_variant.model_family,
              "model_weights": spec_variant.model_weights,
              "model_version": spec_variant.model_version,
              "checkpoint_id": spec_variant.checkpoint_id,
              "model_local_path": (
                str(spec_variant.model_local_path) if spec_variant.model_local_path else None
              ),
              "checkpoint_registry_path": (
                str(spec_variant.checkpoint_registry_path)
                if spec_variant.checkpoint_registry_path
                else None
              ),
              "ligand_conditioning": ligand_on,
              "sidechain_conditioning": sidechain_on,
              "temperature": list(spec_variant.temperature),
              "backbone_noise": list(spec_variant.backbone_noise),
              "batch_size": spec_variant.batch_size,
              "samples_chunk_size": sample_count,
              "campaign_mode": True,
              "return_logits": False,
              "grid_mode": True,
              "job_id": row["job_id"],
              "chunk_id": row["chunk_id"],
              "sample_start": row["sample_start"],
              "sample_count": row["sample_count"],
              "output_h5_path": row["output_h5_path"],
            }
            rows.append(row)
          job_index += 1

  validate_manifest_rows(rows, required_fixed_policies=fixed_policies)
  return rows


def write_campaign_manifest(
  *,
  base_spec: SamplingSpecification,
  campaign_id: str,
  manifest_path: str | Path,
  designs_per_library_type: int,
  samples_chunk_size: int,
  output_root: str | Path,
  fixed_policies: tuple[str, ...] = ("catalytic_triad", "active_site"),
  state_weight_profiles: tuple[str, ...] = ("equal",),
  planner_version: str = "planner_v1",
  dataset_fingerprint: str = "unknown",
  environment_image: str = "unknown",
  git_sha: str = "unknown",
  config_hash: str = "unknown",
) -> Path:
  """Plan rows and write campaign manifest JSON."""
  rows = plan_campaign_manifest(
    base_spec=base_spec,
    campaign_id=campaign_id,
    designs_per_library_type=designs_per_library_type,
    samples_chunk_size=samples_chunk_size,
    output_root=output_root,
    fixed_policies=fixed_policies,
    state_weight_profiles=state_weight_profiles,
    planner_version=planner_version,
    dataset_fingerprint=dataset_fingerprint,
    environment_image=environment_image,
    git_sha=git_sha,
    config_hash=config_hash,
  )
  metadata = {
    "campaign_id": campaign_id,
    "planner_version": planner_version,
    "designs_per_library_type": designs_per_library_type,
    "samples_chunk_size": samples_chunk_size,
  }
  return write_manifest(manifest_path, rows, metadata=metadata)


def _select_row(
  rows: list[dict[str, Any]],
  *,
  row_index: int | None = None,
  row_hash: str | None = None,
) -> dict[str, Any]:
  if row_hash is not None:
    for row in rows:
      if row.get("manifest_row_hash") == row_hash:
        return row
    msg = f"Manifest row hash not found: {row_hash}"
    raise ValueError(msg)

  resolved_index = 0 if row_index is None else row_index
  if resolved_index < 0 or resolved_index >= len(rows):
    msg = f"Manifest row index out of range: {resolved_index}"
    raise IndexError(msg)
  return rows[resolved_index]


def run_manifest_row(  # noqa: PLR0915
  manifest_path: str | Path,
  *,
  row_index: int | None = None,
  row_hash: str | None = None,
  lock_backend: str = "local_fs",
  distributed_lock_backend: DistributedLockBackend | None = None,
  lock_lease_seconds: int = DEFAULT_LOCK_LEASE_SECONDS,
  heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
  """Execute one manifest row through the standard sampling runtime."""
  payload = load_manifest(manifest_path)
  rows = payload["rows"]
  if not isinstance(rows, list):  # pragma: no cover - guarded by load_manifest
    msg = "Manifest rows payload must be a list."
    raise TypeError(msg)

  row = _select_row(rows, row_index=row_index, row_hash=row_hash)
  sampling_spec_payload = row.get("sampling_spec")
  if not isinstance(sampling_spec_payload, dict):
    msg = "Manifest row must include a 'sampling_spec' object."
    raise TypeError(msg)

  raw_output_path = sampling_spec_payload.get("output_h5_path")
  if raw_output_path is None:
    msg = "Manifest row sampling_spec must include output_h5_path."
    raise ValueError(msg)
  output_h5_path = Path(raw_output_path).resolve()
  output_h5_path.parent.mkdir(parents=True, exist_ok=True)
  lock_path = _lock_path(output_h5_path)
  done_marker_path = _done_marker_path(output_h5_path)
  manifest_hash = str(row["manifest_row_hash"])

  existing_marker = _read_done_marker(done_marker_path)
  if existing_marker is not None:
    _validate_done_marker(
      marker=existing_marker,
      marker_path=done_marker_path,
      output_h5_path=output_h5_path,
      manifest_row_hash=manifest_hash,
    )
    return {
      "status": "already_done",
      "output_h5_path": str(output_h5_path),
      "manifest_row_hash": manifest_hash,
      "done_marker_path": str(done_marker_path),
      "attempt_id": existing_marker.get("attempt_id"),
    }
  if output_h5_path.exists():
    msg = (
      f"Output file already exists without done marker for manifest row {manifest_hash}: "
      f"{output_h5_path}"
    )
    raise RuntimeError(msg)

  attempt_id = uuid.uuid4().hex
  owner_token = f"{manifest_hash}:{attempt_id}"
  partial_path = _partial_output_path(output_h5_path, attempt_id)
  with _campaign_lock_context(
    lock_backend=lock_backend,
    distributed_lock_backend=distributed_lock_backend,
    lock_key_path=lock_path,
    owner_token=owner_token,
    manifest_row_hash=manifest_hash,
    attempt_id=attempt_id,
    lease_seconds=lock_lease_seconds,
    heartbeat_interval_seconds=heartbeat_interval_seconds,
  ) as heartbeat_errors:
    existing_marker = _read_done_marker(done_marker_path)
    if existing_marker is not None:
      _validate_done_marker(
        marker=existing_marker,
        marker_path=done_marker_path,
        output_h5_path=output_h5_path,
        manifest_row_hash=manifest_hash,
      )
      return {
        "status": "already_done",
        "output_h5_path": str(output_h5_path),
        "manifest_row_hash": manifest_hash,
        "done_marker_path": str(done_marker_path),
        "attempt_id": existing_marker.get("attempt_id"),
      }
    if output_h5_path.exists():
      msg = (
        f"Output file already exists without done marker for manifest row {manifest_hash}: "
        f"{output_h5_path}"
      )
      raise RuntimeError(msg)

    worker_payload = dict(sampling_spec_payload)
    worker_payload["output_h5_path"] = str(partial_path)
    sampling_spec = SamplingSpecification(**worker_payload)
    try:
      sample_result = sample(sampling_spec)
      if heartbeat_errors:
        msg = (
          f"Lock heartbeat failed while executing manifest row {manifest_hash}: "
          f"{heartbeat_errors[0]}"
        )
        raise RuntimeError(msg) from heartbeat_errors[0]
      if not partial_path.exists():
        msg = f"Worker did not produce expected partial output file: {partial_path}"
        raise RuntimeError(msg)
      _fsync_file(partial_path)
      artifact_sha256 = _sha256_file(partial_path)
      content_digest_sha256 = _h5_content_digest(partial_path)
      partial_path.replace(output_h5_path)
      _fsync_directory(output_h5_path.parent)
      _write_done_marker(
        marker_path=done_marker_path,
        output_h5_path=output_h5_path,
        manifest_row_hash=manifest_hash,
        attempt_id=attempt_id,
        artifact_sha256=artifact_sha256,
        content_digest_sha256=content_digest_sha256,
        lock_backend=lock_backend,
      )
    finally:
      partial_path.unlink(missing_ok=True)

  result_payload = dict(sample_result) if isinstance(sample_result, dict) else {"sample_result": sample_result}
  result_payload.update(
    {
      "status": str(result_payload.get("status", "completed")),
      "output_h5_path": str(output_h5_path),
      "manifest_row_hash": manifest_hash,
      "attempt_id": attempt_id,
      "done_marker_path": str(done_marker_path),
      "lock_backend": lock_backend,
    },
  )
  return result_payload


def _manifest_rows(manifest_path: str | Path) -> list[dict[str, Any]]:
  payload = load_manifest(manifest_path)
  rows = payload["rows"]
  if not isinstance(rows, list):  # pragma: no cover - guarded by load_manifest
    msg = "Manifest rows payload must be a list."
    raise TypeError(msg)
  normalized_rows: list[dict[str, Any]] = []
  for row in rows:
    if not isinstance(row, dict):  # pragma: no cover - guarded by load_manifest
      msg = "Manifest rows must be JSON objects."
      raise TypeError(msg)
    normalized_rows.append(row)
  return normalized_rows


def execute_manifest(
  manifest_path: str | Path,
  *,
  row_hashes: Sequence[str] | None = None,
  continue_on_error: bool = False,
  lock_backend: str = "local_fs",
  distributed_lock_backend: DistributedLockBackend | None = None,
  lock_lease_seconds: int = DEFAULT_LOCK_LEASE_SECONDS,
  heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> dict[str, Any]:
  """Execute all (or selected) manifest rows and return a structured summary."""
  rows = _manifest_rows(manifest_path)
  requested_hashes = tuple(row_hashes or ())
  requested_hash_set = set(requested_hashes)
  available_hashes = {str(row["manifest_row_hash"]) for row in rows}
  missing_hashes = sorted(requested_hash_set - available_hashes)
  if missing_hashes:
    msg = f"Requested manifest row hashes not found: {missing_hashes}"
    raise ValueError(msg)

  selected_rows = (
    [row for row in rows if str(row["manifest_row_hash"]) in requested_hash_set]
    if requested_hashes
    else rows
  )
  success_statuses = {"ok", "completed", "already_done"}
  row_results: list[dict[str, Any]] = []
  successful_rows = 0
  failed_rows = 0

  for row in selected_rows:
    row_hash = str(row["manifest_row_hash"])
    try:
      result = run_manifest_row(
        manifest_path,
        row_hash=row_hash,
        lock_backend=lock_backend,
        distributed_lock_backend=distributed_lock_backend,
        lock_lease_seconds=lock_lease_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
      )
      row_status = str(result.get("status", "completed"))
      row_results.append(
        {
          "manifest_row_hash": row_hash,
          "status": row_status,
          "output_h5_path": result.get("output_h5_path"),
        },
      )
      if row_status in success_statuses:
        successful_rows += 1
      else:
        failed_rows += 1
    except Exception as exc:
      failed_rows += 1
      row_results.append(
        {
          "manifest_row_hash": row_hash,
          "status": "error",
          "error": str(exc),
        },
      )
      if not continue_on_error:
        msg = f"Execution failed for manifest row {row_hash}: {exc}"
        raise RuntimeError(msg) from exc

  return {
    "schema_version": "campaign_execute_report_v1",
    "manifest_path": str(Path(manifest_path).resolve()),
    "selected_rows": len(selected_rows),
    "successful_rows": successful_rows,
    "failed_rows": failed_rows,
    "row_results": row_results,
  }


def plan_scale_ramp(
  *,
  base_spec: SamplingSpecification,
  campaign_id: str,
  manifest_dir: str | Path,
  output_root: str | Path,
  stage_designs_per_library_type: Sequence[int],
  samples_chunk_size: int,
  fixed_policies: tuple[str, ...] = ("catalytic_triad", "active_site"),
  state_weight_profiles: tuple[str, ...] = ("equal",),
  planner_version: str = "planner_v1",
  dataset_fingerprint: str = "unknown",
  environment_image: str = "unknown",
  git_sha: str = "unknown",
  config_hash: str = "unknown",
) -> dict[str, Any]:
  """Create staged manifests for pilot-to-scale rollout."""
  if not stage_designs_per_library_type:
    msg = "stage_designs_per_library_type must contain at least one stage."
    raise ValueError(msg)
  stage_sizes = tuple(int(size) for size in stage_designs_per_library_type)
  if any(size <= 0 for size in stage_sizes):
    msg = "All stage designs_per_library_type values must be positive."
    raise ValueError(msg)
  if any(current <= previous for previous, current in pairwise(stage_sizes)):
    msg = "Stage sizes must be strictly increasing for ramp planning."
    raise ValueError(msg)

  manifest_dir_path = Path(manifest_dir).resolve()
  output_root_path = Path(output_root).resolve()
  manifest_dir_path.mkdir(parents=True, exist_ok=True)
  output_root_path.mkdir(parents=True, exist_ok=True)

  stages: list[ScaleStageResult] = []
  for stage_index, designs_per_library_type in enumerate(stage_sizes):
    stage_campaign_id = f"{campaign_id}-dpl-{designs_per_library_type}"
    stage_manifest_path = manifest_dir_path / f"{stage_campaign_id}.manifest.json"
    stage_output_root = output_root_path / f"dpl_{designs_per_library_type}"
    write_campaign_manifest(
      base_spec=base_spec,
      campaign_id=stage_campaign_id,
      manifest_path=stage_manifest_path,
      designs_per_library_type=designs_per_library_type,
      samples_chunk_size=samples_chunk_size,
      output_root=stage_output_root,
      fixed_policies=fixed_policies,
      state_weight_profiles=state_weight_profiles,
      planner_version=planner_version,
      dataset_fingerprint=dataset_fingerprint,
      environment_image=environment_image,
      git_sha=git_sha,
      config_hash=config_hash,
    )
    stages.append(
      {
        "stage_index": stage_index,
        "designs_per_library_type": designs_per_library_type,
        "campaign_id": stage_campaign_id,
        "manifest_path": str(stage_manifest_path),
        "output_root": str(stage_output_root),
      },
    )

  return {
    "schema_version": "campaign_scale_ramp_plan_v1",
    "campaign_id": campaign_id,
    "samples_chunk_size": samples_chunk_size,
    "stages": stages,
    "rollback_triggers": [
      "integrity_failure",
      "determinism_drift",
      "resource_envelope_violation",
      "lock_integrity_anomaly",
    ],
  }


def evaluate_scale_ramp_reports(report_paths: Sequence[str | Path]) -> dict[str, Any]:
  """Evaluate staged gate reports and produce promote/rollback recommendation."""
  if not report_paths:
    msg = "At least one gate report path is required for rollout evaluation."
    raise ValueError(msg)

  stage_summaries: list[dict[str, Any]] = []
  rollback_stage_index: int | None = None
  rollback_reasons: set[str] = set()

  for stage_index, report_path in enumerate(report_paths):
    report_payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not isinstance(report_payload, dict):
      msg = f"Gate report at {report_path} must be a JSON object."
      raise TypeError(msg)
    if report_payload.get("schema_version") != "campaign_gate_report_v1":
      msg = (
        f"Unsupported gate report schema at {report_path}: "
        f"{report_payload.get('schema_version')!r}."
      )
      raise ValueError(msg)
    gates = report_payload.get("gates")
    if not isinstance(gates, dict):
      msg = f"Gate report at {report_path} is missing 'gates' object."
      raise TypeError(msg)
    promote = bool(report_payload.get("promote"))
    integrity_failure = not (
      bool(gates.get("zero_crashes"))
      and bool(gates.get("zero_lineage_collisions"))
      and bool(gates.get("metadata_complete"))
      and bool(gates.get("checkpoint_preflight_pass"))
    )
    determinism_drift = not bool(gates.get("determinism_pass"))
    resource_violation = not bool(gates.get("resource_envelope_within_limits"))
    base_payload = report_payload.get("base")
    runtime_issues = (
      base_payload.get("runtime_issues")
      if isinstance(base_payload, dict) and isinstance(base_payload.get("runtime_issues"), list)
      else []
    )
    lock_anomaly = any("lock" in str(issue).lower() for issue in runtime_issues)

    if integrity_failure:
      rollback_reasons.add("integrity_failure")
    if determinism_drift:
      rollback_reasons.add("determinism_drift")
    if resource_violation:
      rollback_reasons.add("resource_envelope_violation")
    if lock_anomaly:
      rollback_reasons.add("lock_integrity_anomaly")

    if not promote and rollback_stage_index is None:
      rollback_stage_index = stage_index

    stage_summaries.append(
      {
        "stage_index": stage_index,
        "report_path": str(Path(report_path).resolve()),
        "manifest_path": report_payload.get("manifest_path"),
        "promote": promote,
        "gate_failures": sorted(
          reason
          for reason, failed in (
            ("integrity_failure", integrity_failure),
            ("determinism_drift", determinism_drift),
            ("resource_envelope_violation", resource_violation),
            ("lock_integrity_anomaly", lock_anomaly),
          )
          if failed
        ),
      },
    )

  promote = rollback_stage_index is None
  return {
    "schema_version": "campaign_scale_ramp_report_v1",
    "promote": promote,
    "rollback_stage_index": rollback_stage_index,
    "rollback_reasons": sorted(rollback_reasons),
    "stages": stage_summaries,
  }


def _row_cell_key(row: dict[str, Any]) -> GateCellKey:
  return (
    str(row["fixed_policy"]),
    str(row["state_weight_profile"]),
    str(row["multi_state_strategy"]),
    tuple(str(value) for value in row["temperature"]),
    tuple(str(value) for value in row["backbone_noise"]),
    bool(row["ligand_conditioning"]),
    bool(row["sidechain_conditioning"]),
  )


def _collect_manifest_gate_state(manifest_path: str | Path) -> ManifestGateState:
  payload = load_manifest(manifest_path)
  rows = payload["rows"]
  if not isinstance(rows, list):  # pragma: no cover - guarded by load_manifest
    msg = "Manifest rows payload must be a list."
    raise TypeError(msg)
  state = ManifestGateState(
    manifest_path=Path(manifest_path).resolve(),
    total_rows=len(rows),
    completed_rows=0,
    digest_by_hash={},
    cell_to_hashes={},
    metadata_issues=[],
    checkpoint_issues=[],
    runtime_issues=[],
  )
  for row in rows:
    if not isinstance(row, dict):  # pragma: no cover - guarded by load_manifest
      state.metadata_issues.append("Encountered non-object manifest row.")
      continue
    manifest_hash = str(row.get("manifest_row_hash", ""))
    if not manifest_hash:
      state.metadata_issues.append("Manifest row missing manifest_row_hash.")
      continue
    sampling_spec_payload = row.get("sampling_spec")
    if not isinstance(sampling_spec_payload, dict):
      state.metadata_issues.append(
        f"Manifest row {manifest_hash} is missing sampling_spec payload.",
      )
      continue
    output_value = sampling_spec_payload.get("output_h5_path")
    if output_value is None:
      state.metadata_issues.append(
        f"Manifest row {manifest_hash} is missing sampling_spec.output_h5_path.",
      )
      continue
    output_h5_path = Path(output_value).resolve()
    done_marker = _read_done_marker(_done_marker_path(output_h5_path))
    if done_marker is None:
      state.metadata_issues.append(
        f"Manifest row {manifest_hash} missing done marker for output {output_h5_path}.",
      )
      continue
    try:
      _validate_done_marker(
        marker=done_marker,
        marker_path=_done_marker_path(output_h5_path),
        output_h5_path=output_h5_path,
        manifest_row_hash=manifest_hash,
      )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
      state.metadata_issues.append(str(exc))
      continue

    state.completed_rows += 1
    state.digest_by_hash[manifest_hash] = str(done_marker["content_digest_sha256"])
    cell_key = _row_cell_key(row)
    state.cell_to_hashes.setdefault(cell_key, set()).add(manifest_hash)

    checkpoint_id = str(row.get("checkpoint_id", "")).strip()
    if not checkpoint_id:
      state.checkpoint_issues.append(f"Manifest row {manifest_hash} has empty checkpoint_id.")

    campaign_mode = bool(sampling_spec_payload.get("campaign_mode"))
    grid_mode = bool(sampling_spec_payload.get("grid_mode"))
    if not campaign_mode or not grid_mode:
      state.runtime_issues.append(
        f"Manifest row {manifest_hash} must run with campaign_mode=True and grid_mode=True.",
      )
    return_logits = bool(sampling_spec_payload.get("return_logits"))
    allow_logits = bool(sampling_spec_payload.get("allow_logits_in_campaign"))
    logits_budget = sampling_spec_payload.get("logits_memory_budget_mb")
    if return_logits and (not allow_logits or logits_budget is None):
      state.runtime_issues.append(
        f"Manifest row {manifest_hash} has invalid campaign logits configuration.",
      )
  return state


def evaluate_campaign_gates(
  manifest_path: str | Path,
  *,
  rerun_manifest_paths: tuple[str | Path, ...] = (),
  require_full_cell_rerun: bool = True,
) -> dict[str, Any]:
  """Evaluate campaign gating status for pilot promotion/rollback decisions."""
  base_state = _collect_manifest_gate_state(manifest_path)
  rerun_reports: list[dict[str, Any]] = []
  mismatched_hashes: set[str] = set()
  full_cell_rerun = False

  for rerun_manifest_path in rerun_manifest_paths:
    rerun_state = _collect_manifest_gate_state(rerun_manifest_path)
    shared_hashes = set(base_state.digest_by_hash).intersection(rerun_state.digest_by_hash)
    matched_hashes = {
      row_hash
      for row_hash in shared_hashes
      if base_state.digest_by_hash[row_hash] == rerun_state.digest_by_hash[row_hash]
    }
    mismatched = shared_hashes - matched_hashes
    mismatched_hashes.update(mismatched)
    if not full_cell_rerun:
      full_cell_rerun = any(
        row_hashes.issubset(matched_hashes) and bool(row_hashes)
        for row_hashes in base_state.cell_to_hashes.values()
      )
    rerun_reports.append(
      {
        "manifest_path": str(rerun_state.manifest_path),
        "total_rows": rerun_state.total_rows,
        "completed_rows": rerun_state.completed_rows,
        "shared_rows": len(shared_hashes),
        "matched_rows": len(matched_hashes),
        "mismatched_rows": len(mismatched),
        "metadata_issues": rerun_state.metadata_issues,
        "checkpoint_issues": rerun_state.checkpoint_issues,
        "runtime_issues": rerun_state.runtime_issues,
      },
    )

  rerun_supplied = bool(rerun_manifest_paths)
  if require_full_cell_rerun:
    determinism_pass = rerun_supplied and not mismatched_hashes and full_cell_rerun
  else:
    determinism_pass = (not mismatched_hashes) and (not rerun_supplied or full_cell_rerun)

  gates = {
    "zero_crashes": base_state.completed_rows == base_state.total_rows,
    "zero_lineage_collisions": True,  # enforced by load_manifest/validate_manifest_rows
    "metadata_complete": len(base_state.metadata_issues) == 0,
    "checkpoint_preflight_pass": len(base_state.checkpoint_issues) == 0,
    "determinism_pass": determinism_pass,
    "resource_envelope_within_limits": len(base_state.runtime_issues) == 0,
  }
  promote = all(gates.values())
  return {
    "schema_version": "campaign_gate_report_v1",
    "manifest_path": str(base_state.manifest_path),
    "promote": promote,
    "gates": gates,
    "base": {
      "total_rows": base_state.total_rows,
      "completed_rows": base_state.completed_rows,
      "metadata_issues": base_state.metadata_issues,
      "checkpoint_issues": base_state.checkpoint_issues,
      "runtime_issues": base_state.runtime_issues,
    },
    "reruns": {
      "required_full_cell_rerun": require_full_cell_rerun,
      "supplied": rerun_supplied,
      "full_cell_rerun": full_cell_rerun,
      "mismatched_row_hashes": sorted(mismatched_hashes),
      "reports": rerun_reports,
    },
  }


def _parse_csv(value: str) -> tuple[str, ...]:
  return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_csv(value: str) -> tuple[int, ...]:
  parsed: list[int] = []
  for item in value.split(","):
    stripped = item.strip()
    if not stripped:
      continue
    parsed.append(int(stripped))
  return tuple(parsed)


def _emit_json(payload: dict[str, Any], output_path: str | None = None) -> None:
  rendered = json.dumps(payload, sort_keys=True, indent=2)
  if output_path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered + "\n", encoding="utf-8")
  sys.stdout.write(rendered + "\n")


def _add_plan_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("plan", help="Generate campaign manifest.")
  parser.add_argument("--inputs", required=True, help="Comma-separated input paths.")
  parser.add_argument("--campaign-id", required=True)
  parser.add_argument("--manifest-path", required=True)
  parser.add_argument("--output-root", required=True)
  parser.add_argument("--designs-per-library-type", type=int, required=True)
  parser.add_argument("--samples-chunk-size", type=int, required=True)
  parser.add_argument("--fixed-policies", default="catalytic_triad,active_site")
  parser.add_argument("--state-weight-profiles", default="equal")


def _add_worker_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("worker", help="Execute one manifest row.")
  parser.add_argument("--manifest-path", required=True)
  parser.add_argument("--row-index", type=int, default=None)
  parser.add_argument("--row-hash", default=None)
  parser.add_argument("--lock-backend", default="local_fs")
  parser.add_argument("--lock-lease-seconds", type=int, default=DEFAULT_LOCK_LEASE_SECONDS)
  parser.add_argument(
    "--heartbeat-interval-seconds",
    type=int,
    default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
  )


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("run", help="Execute manifest rows sequentially.")
  parser.add_argument("--manifest-path", required=True)
  parser.add_argument(
    "--row-hash",
    action="append",
    default=[],
    help="Optional row hash filter; repeat to run specific rows only.",
  )
  parser.add_argument("--continue-on-error", action="store_true")
  parser.add_argument("--summary-path", default=None)
  parser.add_argument("--lock-backend", default="local_fs")
  parser.add_argument("--lock-lease-seconds", type=int, default=DEFAULT_LOCK_LEASE_SECONDS)
  parser.add_argument(
    "--heartbeat-interval-seconds",
    type=int,
    default=DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
  )


def _add_gates_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("gates", help="Evaluate pilot campaign gates.")
  parser.add_argument("--manifest-path", required=True)
  parser.add_argument(
    "--rerun-manifest-path",
    action="append",
    default=[],
    help="Optional rerun manifest path; may be passed multiple times.",
  )
  parser.add_argument("--report-path", default=None)
  parser.add_argument(
    "--allow-missing-rerun",
    action="store_true",
    help="Allow determinism gate without a full-cell rerun.",
  )


def _add_ramp_plan_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("ramp-plan", help="Generate staged ramp manifests.")
  parser.add_argument("--inputs", required=True, help="Comma-separated input paths.")
  parser.add_argument("--campaign-id", required=True)
  parser.add_argument("--manifest-dir", required=True)
  parser.add_argument("--output-root", required=True)
  parser.add_argument("--stage-designs-per-library-type", required=True)
  parser.add_argument("--samples-chunk-size", type=int, required=True)
  parser.add_argument("--fixed-policies", default="catalytic_triad,active_site")
  parser.add_argument("--state-weight-profiles", default="equal")
  parser.add_argument("--plan-path", default=None)


def _add_ramp_eval_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
  parser = subparsers.add_parser("ramp-evaluate", help="Evaluate staged gate reports.")
  parser.add_argument(
    "--report-path",
    action="append",
    required=True,
    help="Stage gate report path; pass in stage order.",
  )
  parser.add_argument("--summary-path", default=None)


def _handle_plan_command(args: argparse.Namespace) -> int:
  base_spec = SamplingSpecification(
    inputs=_parse_csv(args.inputs),
    return_logits=False,
  )
  write_campaign_manifest(
    base_spec=base_spec,
    campaign_id=args.campaign_id,
    manifest_path=args.manifest_path,
    designs_per_library_type=args.designs_per_library_type,
    samples_chunk_size=args.samples_chunk_size,
    output_root=args.output_root,
    fixed_policies=_parse_csv(args.fixed_policies),
    state_weight_profiles=_parse_csv(args.state_weight_profiles),
  )
  return 0


def _handle_worker_command(args: argparse.Namespace) -> int:
  run_manifest_row(
    args.manifest_path,
    row_index=args.row_index,
    row_hash=args.row_hash,
    lock_backend=args.lock_backend,
    lock_lease_seconds=args.lock_lease_seconds,
    heartbeat_interval_seconds=args.heartbeat_interval_seconds,
  )
  return 0


def _handle_run_command(args: argparse.Namespace) -> int:
  summary = execute_manifest(
    args.manifest_path,
    row_hashes=tuple(args.row_hash),
    continue_on_error=args.continue_on_error,
    lock_backend=args.lock_backend,
    lock_lease_seconds=args.lock_lease_seconds,
    heartbeat_interval_seconds=args.heartbeat_interval_seconds,
  )
  _emit_json(summary, args.summary_path)
  return 0 if summary["failed_rows"] == 0 else 1


def _handle_gates_command(args: argparse.Namespace) -> int:
  report = evaluate_campaign_gates(
    args.manifest_path,
    rerun_manifest_paths=tuple(args.rerun_manifest_path),
    require_full_cell_rerun=not args.allow_missing_rerun,
  )
  _emit_json(report, args.report_path)
  return 0 if report["promote"] else 2


def _handle_ramp_plan_command(args: argparse.Namespace) -> int:
  base_spec = SamplingSpecification(
    inputs=_parse_csv(args.inputs),
    return_logits=False,
  )
  plan_payload = plan_scale_ramp(
    base_spec=base_spec,
    campaign_id=args.campaign_id,
    manifest_dir=args.manifest_dir,
    output_root=args.output_root,
    stage_designs_per_library_type=_parse_int_csv(args.stage_designs_per_library_type),
    samples_chunk_size=args.samples_chunk_size,
    fixed_policies=_parse_csv(args.fixed_policies),
    state_weight_profiles=_parse_csv(args.state_weight_profiles),
  )
  _emit_json(plan_payload, args.plan_path)
  return 0


def _handle_ramp_eval_command(args: argparse.Namespace) -> int:
  summary = evaluate_scale_ramp_reports(tuple(args.report_path))
  _emit_json(summary, args.summary_path)
  return 0 if summary["promote"] else 2


def main(argv: list[str] | None = None) -> int:
  """CLI entrypoint for campaign planner and worker operations."""
  parser = argparse.ArgumentParser(description="Campaign planner/worker runner.")
  subparsers = parser.add_subparsers(dest="command", required=True)
  _add_plan_parser(subparsers)
  _add_worker_parser(subparsers)
  _add_run_parser(subparsers)
  _add_gates_parser(subparsers)
  _add_ramp_plan_parser(subparsers)
  _add_ramp_eval_parser(subparsers)

  args = parser.parse_args(argv)
  if args.command == "plan":
    return _handle_plan_command(args)
  if args.command == "worker":
    return _handle_worker_command(args)
  if args.command == "run":
    return _handle_run_command(args)
  if args.command == "gates":
    return _handle_gates_command(args)
  if args.command == "ramp-plan":
    return _handle_ramp_plan_command(args)
  if args.command == "ramp-evaluate":
    return _handle_ramp_eval_command(args)
  msg = f"Unsupported command: {args.command!r}"
  raise ValueError(msg)


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
