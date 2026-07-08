"""Tests for host/campaign.py's Zarr content-digest verification primitives."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import zarr

from aminx.host.campaign import _fsync_tree, _zarr_content_digest


def _make_store(tmp_path: Path, name: str = "test.zarr") -> Path:
  store_path = tmp_path / name
  root = zarr.open_group(str(store_path), mode="a")
  arr = root.create_array(name="data", shape=(3,), dtype="int32")
  arr[...] = np.array([1, 2, 3], dtype=np.int32)
  arr.attrs["label"] = "alpha"
  return store_path


def test_digest_is_deterministic(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  assert _zarr_content_digest(store_path) == _zarr_content_digest(store_path)


def test_digest_changes_when_array_data_changes(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  original = _zarr_content_digest(store_path)

  root = zarr.open_group(str(store_path), mode="a")
  root["data"][...] = np.array([9, 9, 9], dtype=np.int32)

  assert _zarr_content_digest(store_path) != original


def test_digest_changes_when_attrs_change(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  original = _zarr_content_digest(store_path)

  root = zarr.open_group(str(store_path), mode="a")
  root["data"].attrs["label"] = "beta"

  assert _zarr_content_digest(store_path) != original


def test_digest_covers_nested_groups(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  original = _zarr_content_digest(store_path)

  root = zarr.open_group(str(store_path), mode="a")
  sub = root.require_group("nested")
  arr = sub.create_array(name="extra", shape=(1,), dtype="int32")
  arr[...] = np.array([42], dtype=np.int32)

  assert _zarr_content_digest(store_path) != original


def test_digest_stable_across_reopen(tmp_path: Path) -> None:
  """Digest computed by a fresh process/session (new zarr.open_group call) matches."""
  store_path = _make_store(tmp_path)
  first = _zarr_content_digest(store_path)
  second = _zarr_content_digest(store_path)
  assert first == second


def test_fsync_tree_does_not_raise_on_real_store(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  _fsync_tree(store_path)  # should not raise


def test_fsync_tree_syncs_every_file_and_directory(tmp_path: Path) -> None:
  store_path = _make_store(tmp_path)
  root = zarr.open_group(str(store_path), mode="a")
  sub = root.require_group("nested")
  arr = sub.create_array(name="extra", shape=(1,), dtype="int32")
  arr[...] = np.array([1], dtype=np.int32)

  all_files = [p for p in store_path.rglob("*") if p.is_file()]
  all_dirs = [p for p in store_path.rglob("*") if p.is_dir()]
  assert all_files, "fixture should have produced at least one chunk/metadata file"

  with (
    patch("aminx.host.campaign._fsync_file") as mock_fsync_file,
    patch("aminx.host.campaign._fsync_directory") as mock_fsync_dir,
  ):
    _fsync_tree(store_path)
    synced_files = {call.args[0] for call in mock_fsync_file.call_args_list}
    synced_dirs = {call.args[0] for call in mock_fsync_dir.call_args_list}
    assert synced_files == set(all_files)
    assert synced_dirs == {*all_dirs, store_path}
