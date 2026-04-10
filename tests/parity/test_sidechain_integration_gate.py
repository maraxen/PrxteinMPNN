"""Tests for side-chain-conditioned parity integration gate logic."""

from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
COLLECT_MODULE = runpy.run_path(str(ROOT / "scripts" / "collect_parity_evidence.py"))

SidechainMacroAcceptance = COLLECT_MODULE["SidechainMacroAcceptance"]
evaluate_sidechain_gate = COLLECT_MODULE["_evaluate_sidechain_conditioned_gate"]
extract_sidechain_acceptance = COLLECT_MODULE["_extract_sidechain_macro_acceptance"]
sidechain_status_code = COLLECT_MODULE["_sidechain_gate_status_code"]


def test_sidechain_gate_warns_when_lane_not_requested() -> None:
  """Lane should warn when side-chain-conditioned collection is disabled."""
  acceptance = SidechainMacroAcceptance(0.05, 0.08, 0.05, 0.05)
  status, reason = evaluate_sidechain_gate(
    requested=False,
    macro_metrics=None,
    acceptance=acceptance,
  )
  assert status == "warn"
  assert "Excluded by parity case corpus configuration." in reason
  assert sidechain_status_code(status) == 1.0


def test_sidechain_gate_fails_when_thresholds_are_violated() -> None:
  """Gate should fail when explicit side-chain thresholds are exceeded."""
  acceptance = SidechainMacroAcceptance(0.05, 0.08, 0.05, 0.05)
  status, reason = evaluate_sidechain_gate(
    requested=True,
    macro_metrics={
      "macro_identity_wasserstein": 0.09,
      "macro_entropy_wasserstein": 0.02,
      "macro_composition_js_distance": 0.01,
      "macro_identity_ks_pvalue": 0.9,
    },
    acceptance=acceptance,
  )
  assert status == "fail"
  assert "identity_wasserstein" in reason
  assert sidechain_status_code(status) == 2.0


def test_sidechain_gate_passes_when_thresholds_are_met() -> None:
  """Gate should pass when side-chain-conditioned metrics satisfy thresholds."""
  acceptance = SidechainMacroAcceptance(0.05, 0.08, 0.05, 0.05)
  status, reason = evaluate_sidechain_gate(
    requested=True,
    macro_metrics={
      "macro_identity_wasserstein": 0.0,
      "macro_entropy_wasserstein": 0.01,
      "macro_composition_js_distance": 0.0,
      "macro_identity_ks_pvalue": 0.95,
    },
    acceptance=acceptance,
  )
  assert status == "pass"
  assert "acceptance checks passed" in reason
  assert sidechain_status_code(status) == 0.0


def test_sidechain_acceptance_is_loaded_from_matrix_manifest() -> None:
  """Collector should expose side-chain acceptance values from parity matrix."""
  acceptance = extract_sidechain_acceptance()
  assert acceptance.identity_wasserstein_max > 0.0
  assert acceptance.entropy_wasserstein_max > 0.0
  assert acceptance.composition_js_max > 0.0
  assert acceptance.identity_ks_pvalue_min > 0.0
