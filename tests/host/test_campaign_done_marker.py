"""Tests for host/campaign.py's done-marker write/validate round-trip against Zarr stores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from xtrax.run import zarr_content_digest

from aminx.host.campaign import (
  DONE_MARKER_SCHEMA_VERSION,
  _done_marker_path,
  _read_done_marker,
  _validate_done_marker,
  _write_done_marker,
)


def _make_store(tmp_path: Path) -> Path:
  store_path = tmp_path / "output.zarr"
  root = zarr.open_group(str(store_path), mode="a")
  arr = root.create_array(name="data", shape=(2,), dtype="int32")
  arr[...] = np.array([1, 2], dtype=np.int32)
  return store_path


def test_write_then_validate_succeeds(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  marker_path = _done_marker_path(store_path)
  digest = zarr_content_digest(store_path)

  _write_done_marker(
    marker_path=marker_path,
    output_h5_path=store_path,
    manifest_row_hash="hash123",
    attempt_id="attempt1",
    content_digest_sha256=digest,
    lock_backend="local_fs",
  )
  marker = _read_done_marker(marker_path)
  assert marker is not None
  assert "artifact_sha256" not in marker  # dropped -- redundant with content digest

  _validate_done_marker(
    marker=marker,
    marker_path=marker_path,
    output_h5_path=store_path,
    manifest_row_hash="hash123",
  )  # should not raise


def test_validate_rejects_schema_mismatch(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  marker_path = _done_marker_path(store_path)
  bad_marker = {
    "schema_version": "some_old_h5_era_schema",
    "manifest_row_hash": "hash123",
    "content_digest_sha256": zarr_content_digest(store_path),
  }
  with pytest.raises(ValueError, match="schema mismatch"):
    _validate_done_marker(
      marker=bad_marker,
      marker_path=marker_path,
      output_h5_path=store_path,
      manifest_row_hash="hash123",
    )


def test_validate_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  marker_path = _done_marker_path(store_path)
  marker = {
    "schema_version": DONE_MARKER_SCHEMA_VERSION,
    "manifest_row_hash": "different_hash",
    "content_digest_sha256": zarr_content_digest(store_path),
  }
  with pytest.raises(ValueError, match="manifest hash mismatch"):
    _validate_done_marker(
      marker=marker,
      marker_path=marker_path,
      output_h5_path=store_path,
      manifest_row_hash="hash123",
    )


def test_validate_rejects_missing_output(tmp_path: Path) -> None:
  store_path = tmp_path / "missing.zarr"
  marker_path = _done_marker_path(store_path)
  marker = {
    "schema_version": DONE_MARKER_SCHEMA_VERSION,
    "manifest_row_hash": "hash123",
    "content_digest_sha256": "irrelevant",
  }
  with pytest.raises(ValueError, match="output.*missing"):
    _validate_done_marker(
      marker=marker,
      marker_path=marker_path,
      output_h5_path=store_path,
      manifest_row_hash="hash123",
    )


def test_validate_detects_corrupted_content(tmp_path: Path) -> None:
  """Simulates post-completion corruption: array mutated after the marker was written."""
  store_path = _make_store(tmp_path)
  marker_path = _done_marker_path(store_path)
  digest = zarr_content_digest(store_path)
  _write_done_marker(
    marker_path=marker_path,
    output_h5_path=store_path,
    manifest_row_hash="hash123",
    attempt_id="attempt1",
    content_digest_sha256=digest,
    lock_backend="local_fs",
  )
  marker = _read_done_marker(marker_path)
  assert marker is not None

  root = zarr.open_group(str(store_path), mode="a")
  root["data"][...] = np.array([999, 999], dtype=np.int32)

  with pytest.raises(ValueError, match="content digest mismatch"):
    _validate_done_marker(
      marker=marker,
      marker_path=marker_path,
      output_h5_path=store_path,
      manifest_row_hash="hash123",
    )
