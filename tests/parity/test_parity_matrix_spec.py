"""Parity matrix manifest checks."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from prxteinmpnn.parity.matrix import (
  ligand_tied_multistate_enforcement_for_tier,
  ligand_tied_multistate_rollout_outcome,
  ligand_tied_multistate_rollout_policy,
  load_parity_matrix,
)


def test_parity_matrix_contains_required_paths() -> None:
  """Ensure the parity matrix covers the required code paths."""
  paths = load_parity_matrix()
  ids = {path.id for path in paths}
  expected = {
    "protein-feature-extraction",
    "protein-encoder",
    "decoder-unconditional",
    "decoder-conditional-scoring",
    "autoregressive-sampling",
    "tied-positions-and-multi-state",
    "ligand-tied-positions-and-multi-state",
    "logits-helper-branches",
    "ligand-feature-extraction",
    "ligand-conditioning-context",
    "ligand-autoregressive",
    "side-chain-packer",
    "averaged-encoding",
    "end-to-end-run-apis",
    "checkpoint-family-load",
  }
  assert ids == expected


@pytest.mark.parametrize("path_id", ["protein-feature-extraction", "side-chain-packer"])
def test_parity_matrix_has_numeric_thresholds(path_id: str) -> None:
  """Check that critical heavy paths include executable acceptance thresholds."""
  path = next(item for item in load_parity_matrix() if item.id == path_id)
  assert path.acceptance
  assert path.method
  assert path.reference


def test_ligand_sidechain_macro_acceptance_is_explicit() -> None:
  """Ensure side-chain-conditioned macro acceptance criteria are defined."""
  path = next(item for item in load_parity_matrix() if item.id == "ligand-autoregressive")
  acceptance = path.acceptance["sidechain_conditioned_macro"]
  assert isinstance(acceptance, dict)
  for key in (
    "identity_wasserstein_max",
    "entropy_wasserstein_max",
    "composition_js_max",
    "identity_ks_pvalue_min",
  ):
    assert isinstance(acceptance[key], (int, float))


@pytest.mark.parametrize(
  "path_id",
  ["tied-positions-and-multi-state", "ligand-tied-positions-and-multi-state"],
)
def test_tied_multistate_comparison_lanes_are_explicit(path_id: str) -> None:
  """Ensure tied/multistate parity lanes are explicit and uniquely primary."""
  path = next(item for item in load_parity_matrix() if item.id == path_id)
  lanes = path.acceptance["comparison_lanes"]
  assert isinstance(lanes, list)
  assert len(lanes) >= 1

  primary_count = 0
  seen_conditions: set[str] = set()
  for lane in lanes:
    assert isinstance(lane, dict)
    assert isinstance(lane["condition"], str)
    assert lane["condition"] not in seen_conditions
    seen_conditions.add(lane["condition"])
    assert lane["comparison_api"] in {"sampling", "scoring"}
    assert lane["reference_combiner"] in {"weighted_sum", "arithmetic_mean", "geometric_mean"}
    assert lane["jax_multi_state_strategy"] in {"arithmetic_mean", "geometric_mean", "product"}
    assert lane.get("input_context", "ligand_context") in {
      "ligand_context",
      "side_chain_conditioned",
    }
    assert lane["token_comparison"] in {"enabled", "disabled"}
    assert isinstance(lane["is_primary"], bool)
    if lane["is_primary"]:
      primary_count += 1

  assert primary_count == 1


def test_ligand_tied_multistate_rollout_policy_is_staged() -> None:
  """Ensure ligand tied/multistate rollout policy is warn-heavy and fail-audit."""
  policy = ligand_tied_multistate_rollout_policy()
  assert policy["parity_heavy"] == "warn"
  assert policy["parity_audit"] == "fail"
  assert ligand_tied_multistate_enforcement_for_tier("parity_heavy") == "warn"
  assert ligand_tied_multistate_enforcement_for_tier("parity_audit") == "fail"
  assert ligand_tied_multistate_enforcement_for_tier("unknown-tier") == "fail"
  assert ligand_tied_multistate_rollout_outcome(condition_passed=True, tier="parity_heavy") == "pass"
  assert (
    ligand_tied_multistate_rollout_outcome(condition_passed=False, tier="parity_heavy") == "warn"
  )
  assert (
    ligand_tied_multistate_rollout_outcome(condition_passed=False, tier="parity_audit") == "fail"
  )


def test_parity_matrix_manifest_is_valid_json() -> None:
  """Ensure the manifest file remains parseable as JSON."""
  manifest_path = Path(__file__).with_name("parity_matrix.json")
  payload = json.loads(manifest_path.read_text(encoding="utf-8"))
  assert payload["version"] == 1
  assert isinstance(payload["paths"], list)
