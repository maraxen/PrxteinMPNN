"""Tests for aminx.io.designs.DesignZarrWriter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from aminx.io.designs import DesignMetadata, DesignPayload, DesignZarrWriter

N_CANONICAL = 4
N_STATES = 2


def _payload(seq_value: int = 1) -> DesignPayload:
  metadata: DesignMetadata = {
    "pool_type": "BackboneOnly",
    "state_mapping": [0, 1],
    "weight_strategy": "uniform",
    "combination_algorithm": "none",
    "structure_ids": ["struct_0"],
    "parent_structure_idx": 0,
  }
  return {
    "sequence": np.full(N_CANONICAL, seq_value, dtype=np.uint8),
    "logits": np.zeros((N_CANONICAL, 21), dtype=np.float32),
    "scores": np.array([0.5], dtype=np.float32),
    "state_weights": np.ones(N_STATES, dtype=np.float32) / N_STATES,
    "metadata": metadata,
  }


def _writer(tmp_path: Path) -> DesignZarrWriter:
  return DesignZarrWriter.from_multistate_shapes(
    str(tmp_path / "designs.zarr"),
    n_canonical=N_CANONICAL,
    n_states=N_STATES,
  )


def test_write_and_close_persists_to_zarr(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  writer.write((0,), _payload(seq_value=7))
  writer.close()

  root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
  group = root["0"]
  assert np.array_equal(group["sequence"][:], np.full(N_CANONICAL, 7, dtype=np.uint8))
  assert group["logits"].dtype == np.float16
  assert group["scores"][:] == pytest.approx([0.5])
  assert group.attrs["pool_type"] == "BackboneOnly"
  assert group.attrs["parent_structure_idx"] == 0


def test_context_manager_drains_on_exit(tmp_path: Path) -> None:
  with _writer(tmp_path) as writer:
    writer.write((0,), _payload())

  root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
  assert "0" in root


def test_multiple_designs_get_distinct_keys(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  writer.write((0,), _payload(seq_value=1))
  writer.write((1,), _payload(seq_value=2))
  writer.close()

  root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
  assert np.array_equal(root["0"]["sequence"][:], np.full(N_CANONICAL, 1, dtype=np.uint8))
  assert np.array_equal(root["1"]["sequence"][:], np.full(N_CANONICAL, 2, dtype=np.uint8))


def test_wrong_sequence_shape_raises(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  bad = _payload()
  bad["sequence"] = np.zeros(N_CANONICAL + 1, dtype=np.uint8)
  with pytest.raises(AssertionError, match="sequence shape"):
    writer.write((0,), bad)


def test_out_of_range_logits_raise(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  bad = _payload()
  bad["logits"] = np.full((N_CANONICAL, 21), 1e5, dtype=np.float32)
  with pytest.raises(AssertionError, match="float16-safe range"):
    writer.write((0,), bad)


def test_nested_structure_key_becomes_zarr_group_path(tmp_path: Path) -> None:
  writer = _writer(tmp_path)
  writer.write(("structure_3", "sample_0"), _payload())
  writer.close()

  root = zarr.open_group(str(tmp_path / "designs.zarr"), mode="r")
  assert "sequence" in root["structure_3"]["sample_0"]
