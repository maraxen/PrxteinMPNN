"""Synthetic-input checks of the X11b methods gate: the outcome ladder, the comparator, fragility, and the planted controls.

Every negative here must be caught: a sequence or pin change, an energy past the 1e-4 bar, a fragile pick that is not exempt, a control transform
that changes nothing or mutates its input. The JAX and Node arms are not exercised here (remote only).
"""

from __future__ import annotations

import copy
from typing import NamedTuple

import numpy as np
import pytest

from scripts.browser_validation import protonpotts_ph_methods_node_gate as gate


class _Event(NamedTuple):
  probs: np.ndarray
  uniform: float
  choice: int


def _design(**over: object) -> dict:
  base = {
    "sample": 0,
    "sequence": [1, 2, 3, 4],
    "designable": [2, 3],
    "pins": [[1, 7, 5]],
    "label": "",
    "n_draws": 6,
    "final_potts_energy": -10.0,
    "selective_energy": 0.5,
    "choices": [2, 3, 1],
    "first_fragile": None,
  }
  base.update(over)
  return base


def _good() -> dict:
  return {
    "jax_arm_ok": True, "artifacts_verified": True, "node_ok": True, "reference_consistent": True,
    "control_tokens_detected": True, "control_randint_detected": True, "control_perm_detected": True,
    "control_energy_detected": True, "n_cases": 20, "n_pass": 20,
  }  # fmt: skip


# ---- outcome ladder -------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["jax_arm_ok", "artifacts_verified", "node_ok", "reference_consistent"])
def test_error_outranks_everything(key: str) -> None:
  result = _good()
  result[key] = False
  assert gate._decide(result) == "error"


def test_missing_error_key_is_error() -> None:
  result = _good()
  del result["node_ok"]
  assert gate._decide(result) == "error"


@pytest.mark.parametrize("control", ["control_tokens_detected", "control_randint_detected", "control_perm_detected", "control_energy_detected"])
def test_any_missed_control_is_instrument_unverified(control: str) -> None:
  result = _good()
  result[control] = False
  assert gate._decide(result) == "instrument_unverified"


def test_instrument_unverified_outranks_a_fail() -> None:
  result = _good()
  result["control_perm_detected"] = False
  result["n_pass"] = 3
  assert gate._decide(result) == "instrument_unverified"


def test_pass_needs_every_case() -> None:
  assert gate._decide(_good()) == "pass"
  result = _good()
  result["n_pass"] = 19
  assert gate._decide(result) == "fail"


def test_zero_cases_is_fail_not_pass() -> None:
  result = _good()
  result["n_cases"] = 0
  result["n_pass"] = 0
  assert gate._decide(result) == "fail"


# ---- comparator -------------------------------------------------------------------------------------------------------------------------------


def test_identical_designs_pass() -> None:
  out = gate.compare_designs([_design()], [_design()])
  assert out == {"pass": True, "fragile": 0, "mismatches": []}


def test_energy_at_the_bar_passes_and_past_it_fails() -> None:
  # ref 0.0 -> denominator max(1, 0) = 1, so the bar is an absolute 1e-4 here
  assert gate.compare_designs([_design(final_potts_energy=1e-4)], [_design(final_potts_energy=0.0)])["pass"]
  out = gate.compare_designs([_design(final_potts_energy=1.5e-4)], [_design(final_potts_energy=0.0)])
  assert not out["pass"]
  assert out["mismatches"][0]["field"] == "final_potts_energy"


def test_energy_bar_is_relative_for_large_references() -> None:
  ref = _design(final_potts_energy=-1000.0)
  assert gate.compare_designs([_design(final_potts_energy=-1000.0 + 0.05)], [ref])["pass"]  # 5e-5 relative
  assert not gate.compare_designs([_design(final_potts_energy=-1000.0 + 0.5)], [ref])["pass"]  # 5e-4 relative


def test_selective_energy_is_graded_too() -> None:
  out = gate.compare_designs([_design(selective_energy=0.6)], [_design(selective_energy=0.5)])
  assert not out["pass"]
  assert out["mismatches"][0]["field"] == "selective_energy"


def test_sequence_mismatch_without_fragility_fails() -> None:
  out = gate.compare_designs([_design(sequence=[1, 2, 3, 9])], [_design()])
  assert not out["pass"]
  assert out["mismatches"][0]["field"] == "sequence"
  assert out["mismatches"][0]["first_pick_difference"] is None


def test_pin_mismatch_fails_even_when_sequences_agree() -> None:
  out = gate.compare_designs([_design(pins=[[1, 8, 5]])], [_design()])
  assert not out["pass"]
  assert "pins" in out["mismatches"][0]["field"]


def test_pin_residue_number_mismatch_fails() -> None:
  out = gate.compare_designs([_design(pins=[[1, 7, 6]])], [_design()])
  assert not out["pass"]


def test_designable_and_label_are_exact() -> None:
  assert not gate.compare_designs([_design(designable=[2])], [_design()])["pass"]
  assert not gate.compare_designs([_design(label="x")], [_design()])["pass"]


def test_n_draws_mismatch_fails() -> None:
  out = gate.compare_designs([_design(n_draws=7)], [_design()])
  assert not out["pass"]
  assert out["mismatches"][0]["field"] == "n_draws"


def test_count_mismatch_fails() -> None:
  assert not gate.compare_designs([_design()], [_design(), _design(sample=1)])["pass"]
  assert not gate.compare_designs([], [])["pass"]


def test_fragile_exemption_applies_when_first_difference_is_at_or_after_first_fragile() -> None:
  ref = _design(sequence=[1, 2, 3, 4], choices=[2, 3, 1], first_fragile=1)
  js = _design(sequence=[1, 2, 3, 9], choices=[2, 3, 5], n_draws=99, final_potts_energy=123.0)  # first difference at 2 >= 1
  out = gate.compare_designs([js], [ref])
  assert out == {"pass": True, "fragile": 1, "mismatches": []}


def test_fragile_exemption_applies_at_exactly_the_fragile_pick() -> None:
  ref = _design(choices=[2, 3, 1], first_fragile=2)
  js = _design(sequence=[1, 2, 3, 9], choices=[2, 3, 5])
  assert gate.compare_designs([js], [ref])["fragile"] == 1


def test_fragile_exemption_does_not_apply_before_the_fragile_pick() -> None:
  ref = _design(choices=[2, 3, 1], first_fragile=2)
  js = _design(sequence=[1, 9, 3, 4], choices=[2, 5, 1])  # first difference at 1 < 2
  out = gate.compare_designs([js], [ref])
  assert not out["pass"]
  assert out["fragile"] == 0


def test_fragile_exemption_does_not_apply_without_a_fragile_pick() -> None:
  ref = _design(choices=[2, 3, 1], first_fragile=None)
  js = _design(sequence=[1, 2, 3, 9], choices=[2, 3, 5])
  assert not gate.compare_designs([js], [ref])["pass"]


def test_exemption_can_be_disabled_for_controls() -> None:
  ref = _design(choices=[2, 3, 1], first_fragile=0)
  js = _design(sequence=[1, 2, 3, 9], choices=[2, 3, 5])
  assert gate.compare_designs([js], [ref])["pass"]
  out = gate.compare_designs([js], [ref], exempt_fragile=False)
  assert not out["pass"]
  assert out["fragile"] == 0


def test_fragile_exemption_never_excuses_a_plan_field() -> None:
  ref = _design(choices=[2, 3, 1], first_fragile=0)
  js = _design(sequence=[1, 2, 3, 9], choices=[2, 3, 5], pins=[[1, 8, 5]])
  out = gate.compare_designs([js], [ref])
  assert not out["pass"]
  assert out["fragile"] == 1


def test_first_pick_difference_handles_short_streams() -> None:
  assert gate.first_pick_difference([1, 2, 3], [1, 2, 3]) is None
  assert gate.first_pick_difference([1, 2, 3], [1, 9, 3]) == 1
  assert gate.first_pick_difference([1, 2], [1, 2, 3]) == 2


# ---- fragility --------------------------------------------------------------------------------------------------------------------------------


def test_pick_on_a_boundary_is_fragile() -> None:
  probs = np.array([0.5, 0.5])
  assert gate.is_fragile_pick(probs, 0.5, None)
  assert gate.is_fragile_pick(probs, 0.5 + 5e-6, None)  # 5e-6 of the total mass, inside 1e-5


def test_pick_far_from_a_boundary_is_not_fragile() -> None:
  probs = np.array([0.5, 0.5])
  assert not gate.is_fragile_pick(probs, 0.2, None)
  assert not gate.is_fragile_pick(probs, 0.5 + 5e-5, None)  # 5e-5 is outside the 1e-5 margin


def test_fragility_follows_the_cdf_order() -> None:
  probs = np.array([0.1, 0.9])
  order = np.array([1, 0])  # cumulative in order: 0.9, 1.0
  assert gate.is_fragile_pick(probs, 0.9, order)
  assert not gate.is_fragile_pick(probs, 0.9, None)  # identity order: boundary at 0.1


def test_non_finite_uniform_is_never_fragile() -> None:
  assert not gate.is_fragile_pick(np.array([0.5, 0.5]), float("nan"), None)


def test_first_fragile_index_reports_the_first_boundary_pick() -> None:
  events = [_Event(np.array([0.5, 0.5]), 0.2, 0), _Event(np.array([0.5, 0.5]), 0.5, 1), _Event(np.array([0.5, 0.5]), 0.5, 0)]
  assert gate.first_fragile_index(events, None) == 1
  assert gate.first_fragile_index(events[:1], None) is None


# ---- controls ---------------------------------------------------------------------------------------------------------------------------------


def _replay() -> list[dict]:
  return [
    {"uniforms": [0.1, 0.4, 0.9], "randints": [0, 1, 2], "perms": [[0, 1, 2]]},
    {"uniforms": [0.2, 0.5, 0.7], "randints": [2, 1, 0], "perms": [[2, 0, 1]]},
  ]


@pytest.mark.parametrize(
  ("transform", "field"),
  [(gate.roll_uniforms, "uniforms"), (gate.roll_randints, "randints"), (gate.reverse_perms, "perms")],
)
def test_control_transform_changes_the_replay_and_leaves_the_original(transform, field: str) -> None:  # noqa: ANN001
  original = _replay()
  snapshot = copy.deepcopy(original)
  planted = transform(original)
  assert original == snapshot
  assert planted is not original
  assert [e[field] for e in planted] != [e[field] for e in original]


def test_roll_uniforms_rolls_each_sample_by_one() -> None:
  planted = gate.roll_uniforms(_replay())
  assert planted[0]["uniforms"] == [0.9, 0.1, 0.4]
  assert planted[1]["uniforms"] == [0.7, 0.2, 0.5]


def test_roll_randints_rolls_by_one() -> None:
  assert gate.roll_randints(_replay())[0]["randints"] == [2, 0, 1]


def test_reverse_perms_reverses_each_permutation() -> None:
  assert gate.reverse_perms(_replay())[1]["perms"] == [[1, 0, 2]]


def test_roll_is_not_applied_to_a_single_randint() -> None:
  one = [{"uniforms": [], "randints": [5], "perms": []}]
  assert gate.roll_randints(one)[0]["randints"] == [5]


def test_control_applicability() -> None:
  replay = _replay()
  assert gate.control_applies("tokens", "converged_mcmc", 0.05, replay)
  assert not gate.control_applies("tokens", "converged_mcmc", 0.0, [{"uniforms": [], "randints": [], "perms": []}])
  assert gate.control_applies("randint", "two_phase", 0.05, replay)
  assert not gate.control_applies("randint", "gibbs", 0.05, replay)
  assert gate.control_applies("perm", "gibbs", 0.05, [{"uniforms": [0.1], "randints": [], "perms": [[1, 0]]}])
  assert not gate.control_applies("perm", "converged_mcmc", 0.05, replay)
  assert gate.control_applies("energy", "gibbs", 0.0, replay)


def test_energy_control_keeps_sequences_and_replaces_energies() -> None:
  ref = [_design(final_potts_energy=-1.0), _design(sample=1, final_potts_energy=-2.0)]
  planted = gate.energy_control_reference(ref, np.array([-1.5, -2.5]))
  assert [r["final_potts_energy"] for r in planted] == [-1.5, -2.5]
  assert [r["sequence"] for r in planted] == [r["sequence"] for r in ref]
  assert ref[0]["final_potts_energy"] == -1.0  # untouched
  flagged = gate.compare_designs([_design(final_potts_energy=-1.0)], planted[:1], exempt_fragile=False)
  assert not flagged["pass"]


def test_tokens_control_flags_a_rolled_reference_against_the_original_js_stream() -> None:
  js = [_design(sequence=[1, 2, 3, 4], choices=[2, 3, 1])]
  rolled_ref = [_design(sequence=[1, 2, 3, 5], choices=[2, 3, 1])]
  assert not gate.compare_designs(js, rolled_ref, exempt_fragile=False)["pass"]


# ---- normalisation ----------------------------------------------------------------------------------------------------------------------------


def test_normalize_js_maps_pin_fields_and_choices() -> None:
  raw = {
    "sample": 1, "sequence": [1, 2], "designable": [0, 1], "label": "",
    "pins": [{"position": 3, "protIdx": 7, "depIdxs": [4], "resId": 9, "type": "HIS-P"}],
    "n_draws": 2, "final_potts_energy": -1.0, "selective_energy": 0.25,
    "events": [{"uniform": 0.3, "choice": 5}, {"uniform": 0.9, "choice": 6}],
  }  # fmt: skip
  got = gate.normalize_js(raw)
  assert got["pins"] == [[3, 7, 9]]
  assert got["choices"] == [5, 6]
  assert got["sample"] == 1
