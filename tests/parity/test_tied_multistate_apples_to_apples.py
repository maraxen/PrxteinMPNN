"""Tests for tied/multistate apples-to-apples alignment helpers."""

from __future__ import annotations

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(str(ROOT / "scripts" / "collect_parity_evidence.py"))
combine_reference_tied_log_probs = MODULE["_combine_reference_tied_log_probs"]
extract_tied_multistate_lanes = MODULE["_extract_tied_multistate_lanes"]


def _row_log_softmax(logits: np.ndarray) -> np.ndarray:
  shifted = logits - np.max(logits, axis=-1, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def test_weighted_sum_lane_combiner_broadcasts_group_log_probs() -> None:
  """Weighted-sum lane aligns tied positions and leaves untied positions unchanged."""
  reference_log_probs = np.log(
    np.array(
      [
        [0.70, 0.20, 0.10],
        [0.40, 0.50, 0.10],
        [0.10, 0.20, 0.70],
      ],
      dtype=np.float64,
    ),
  )
  combined = combine_reference_tied_log_probs(
    reference_log_probs,
    tie_groups=[[0, 1]],
    tie_weights=[[1.0, 2.0]],
    combiner="weighted_sum",
  )
  expected_group = _row_log_softmax(
    np.sum(reference_log_probs[[0, 1]] * np.array([[1.0], [2.0]]), axis=0, keepdims=True),
  )[0]
  np.testing.assert_allclose(combined[0], expected_group, atol=1e-6, rtol=1e-6)
  np.testing.assert_allclose(combined[1], expected_group, atol=1e-6, rtol=1e-6)
  np.testing.assert_allclose(combined[2], reference_log_probs[2], atol=1e-6, rtol=1e-6)


def test_arithmetic_mean_lane_combiner_matches_probability_mean() -> None:
  """Arithmetic-mean lane combines per-position probabilities before logging."""
  reference_log_probs = np.log(
    np.array(
      [
        [0.60, 0.30, 0.10],
        [0.20, 0.70, 0.10],
        [0.25, 0.25, 0.50],
      ],
      dtype=np.float64,
    ),
  )
  combined = combine_reference_tied_log_probs(
    reference_log_probs,
    tie_groups=[[0, 1]],
    tie_weights=[[1.0, 3.0]],
    combiner="arithmetic_mean",
  )
  probs = np.exp(reference_log_probs[[0, 1]])
  expected_prob = (probs[0] * 1.0 + probs[1] * 3.0) / 4.0
  expected_group = np.log(expected_prob)
  np.testing.assert_allclose(combined[0], expected_group, atol=1e-6, rtol=1e-6)
  np.testing.assert_allclose(combined[1], expected_group, atol=1e-6, rtol=1e-6)
  np.testing.assert_allclose(combined[2], reference_log_probs[2], atol=1e-6, rtol=1e-6)


def test_tied_multistate_lanes_have_unique_primary_lane() -> None:
  """Manifest-backed tied lane config has one primary apples-to-apples lane."""
  lanes = extract_tied_multistate_lanes()
  core_lanes = [lane for lane in lanes if lane.path_id == "tied-positions-and-multi-state"]
  ligand_lanes = [lane for lane in lanes if lane.path_id == "ligand-tied-positions-and-multi-state"]

  assert len(core_lanes) == 2
  core_conditions = {lane.condition for lane in core_lanes}
  assert core_conditions == {
    "reference_weighted_sum__jax_product",
    "reference_arithmetic_mean__jax_arithmetic_mean",
  }
  core_primary_lanes = [lane for lane in core_lanes if lane.is_primary]
  assert len(core_primary_lanes) == 1
  assert core_primary_lanes[0].condition == "reference_weighted_sum__jax_product"
  assert core_primary_lanes[0].comparison_api == "sampling"
  assert core_primary_lanes[0].token_comparison_enabled

  assert len(ligand_lanes) == 4
  ligand_primary_lanes = [lane for lane in ligand_lanes if lane.is_primary]
  assert len(ligand_primary_lanes) == 1
  assert ligand_primary_lanes[0].input_context == "ligand_context"
  assert ligand_primary_lanes[0].comparison_api == "sampling"
  assert any(lane.input_context == "side_chain_conditioned" for lane in ligand_lanes)


def test_tied_multistate_lanes_reject_non_equivalent_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
  """Lane extraction rejects non-equivalent reference/JAX strategy mappings."""
  lane = {
    "condition": "invalid-sampling-arithmetic",
    "comparison_api": "sampling",
    "reference_combiner": "arithmetic_mean",
    "jax_multi_state_strategy": "arithmetic_mean",
    "token_comparison": "disabled",
    "is_primary": True,
  }
  fake_path = SimpleNamespace(
    id="tied-positions-and-multi-state",
    acceptance={"comparison_lanes": [lane]},
  )
  monkeypatch.setitem(
    extract_tied_multistate_lanes.__globals__,
    "load_parity_matrix",
    lambda: (fake_path,),
  )

  with pytest.raises(ValueError, match="non-equivalent tied lane mapping"):
    extract_tied_multistate_lanes()
