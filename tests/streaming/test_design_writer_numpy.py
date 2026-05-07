"""Host staging for ``DesignArrayRecordWriter`` (Phase 5g follow-on)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.io.designs import DesignArrayRecordWriter, DesignMetadata


def _minimal_metadata(*, n_states: int = 2) -> DesignMetadata:
  return {
    "pool_type": "BackboneOnly",
    "state_mapping": list(range(n_states)),
    "weight_strategy": "uniform",
    "combination_algorithm": "none",
    "structure_ids": ["s0"],
    "parent_structure_idx": 0,
  }


def test_write_numpy_payload_skips_device_get(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  device_get_calls = 0
  real_device_get = jax.device_get

  def counting_device_get(x: object) -> object:
    nonlocal device_get_calls
    device_get_calls += 1
    return real_device_get(x)

  monkeypatch.setattr(jax, "device_get", counting_device_get)

  out = tmp_path / "designs.arrayrecord"
  writer = DesignArrayRecordWriter(str(out), n_canonical=3, n_states=2)
  payload = {
    "sequence": np.arange(3, dtype=np.uint8),
    "logits": np.zeros((3, 21), dtype=np.float32),
    "scores": np.array([0.0], dtype=np.float32),
    "state_weights": np.ones(2, dtype=np.float32) / 2.0,
    "metadata": _minimal_metadata(),
  }
  try:
    writer.write(payload)
  finally:
    writer.close()

  assert device_get_calls == 0


def test_write_jax_array_calls_device_get(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  mock_dg = MagicMock(side_effect=jax.device_get)
  monkeypatch.setattr(jax, "device_get", mock_dg)

  out = tmp_path / "designs.arrayrecord"
  writer = DesignArrayRecordWriter(str(out), n_canonical=3, n_states=2)
  payload = {
    "sequence": np.arange(3, dtype=np.uint8),
    "logits": jnp.zeros((3, 21), dtype=jnp.float32),
    "scores": np.array([0.0], dtype=np.float32),
    "state_weights": np.ones(2, dtype=np.float32) / 2.0,
    "metadata": _minimal_metadata(),
  }
  try:
    writer.write(payload)
  finally:
    writer.close()

  assert mock_dg.call_count >= 1
