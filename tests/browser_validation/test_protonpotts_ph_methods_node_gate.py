"""Synthetic-input checks of the X11b methods gate: the outcome ladder, the comparator, the probability bound, and the planted controls.

Every negative here must be caught: a sequence or pin change, an energy past the 1e-4 bar, an aligned-pick discrepancy past PROB_BOUND, a
divergence the measured discrepancy does not explain, an exemption that reaches a plan field, a control transform that changes nothing or
mutates its input. The JAX and Node arms are not exercised here (remote only).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
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
    "choices": [0, 0, 0],
    "probs": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
    "uniforms": [0.1, 0.5, 0.9],
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


# ---- comparator: plan, energies, sequence -----------------------------------------------------------------------------------------------------


def test_identical_designs_pass() -> None:
  out = gate.compare_designs([_design()], [_design()])
  assert out["pass"] is True
  assert out["fragile"] == 0
  assert out["mismatches"] == []
  assert out["max_discrepancy"] == 0.0


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


# ---- explained divergence (amendment 261010) --------------------------------------------------------------------------------------------------


def test_divergence_explained_by_measured_discrepancy_is_exempt() -> None:
  # Pick 1: the reference u = 0.5 sits exactly on a cumulative boundary (boundary distance 0). JS draws from a distribution 1e-4 away, so the
  # measured discrepancy (1e-4) covers the boundary distance (0): the first difference is exempt, reported as fragile, and the design passes.
  ref = _design(uniforms=[0.1, 0.5, 0.9])
  js = _design(sequence=[1, 2, 3, 9], choices=[0, 1, 0], probs=[[0.5, 0.5], [0.5 + 1e-4, 0.5 - 1e-4], [0.5, 0.5]],
               n_draws=99, final_potts_energy=123.0)  # fmt: skip
  out = gate.compare_designs([js], [ref])
  assert out["pass"]
  assert out["fragile"] == 1
  assert out["mismatches"] == []
  assert out["max_discrepancy"] == 0.0  # the only aligned pick (pick 0) is identical


def test_divergence_exactly_on_the_boundary_is_exempt_with_zero_discrepancy() -> None:
  ref = _design(uniforms=[0.1, 0.5, 0.9])
  js = _design(sequence=[1, 2, 3, 9], choices=[0, 1, 0])  # identical probs: discrepancy 0 >= boundary distance 0
  out = gate.compare_designs([js], [ref])
  assert out["pass"]
  assert out["fragile"] == 1


def test_divergence_larger_than_the_boundary_distance_fails_naming_the_sequence() -> None:
  # Pick 1: u = 0.2 is 0.3 of the mass from the nearest boundary (0.5); a 1e-4 discrepancy does not explain a different bin.
  ref = _design(uniforms=[0.1, 0.2, 0.9])
  js = _design(sequence=[1, 2, 3, 9], choices=[0, 1, 0], probs=[[0.5, 0.5], [0.5 + 1e-4, 0.5 - 1e-4], [0.5, 0.5]])
  out = gate.compare_designs([js], [ref])
  assert not out["pass"]
  assert out["fragile"] == 0
  assert [m["field"] for m in out["mismatches"]] == ["sequence"]
  assert out["mismatches"][0]["first_pick_difference"] == 1


def test_aligned_discrepancy_past_the_bound_fails_even_when_sequences_agree() -> None:
  js = _design(probs=[[0.5 + 2e-3, 0.5 - 2e-3], [0.5, 0.5], [0.5, 0.5]])  # pick 0 is aligned: 2e-3 > 1e-3
  for exempt in (True, False):
    out = gate.compare_designs([js], [_design()], exempt_fragile=exempt)
    assert not out["pass"]
    assert [m["field"] for m in out["mismatches"]] == ["prob_bound"]
    assert out["mismatches"][0]["max_discrepancy"] == pytest.approx(2e-3)
    assert out["max_discrepancy"] == pytest.approx(2e-3)


def test_aligned_discrepancy_within_the_bound_passes() -> None:
  js = _design(probs=[[0.5 + 5e-4, 0.5 - 5e-4], [0.5, 0.5], [0.5, 0.5]])
  out = gate.compare_designs([js], [_design()])
  assert out["pass"]
  assert out["max_discrepancy"] == pytest.approx(5e-4)


def test_prob_bound_covers_only_the_aligned_picks_before_a_divergence() -> None:
  # Pick 1 diverges. Its own discrepancy is not an aligned pick, so it is judged only through the explained-divergence rule (and fails here).
  ref = _design(uniforms=[0.1, 0.2, 0.9])
  js = _design(sequence=[1, 2, 3, 9], choices=[0, 1, 0], probs=[[0.5, 0.5], [0.5 + 5e-3, 0.5 - 5e-3], [0.5, 0.5]])
  out = gate.compare_designs([js], [ref])
  assert [m["field"] for m in out["mismatches"]] == ["sequence"]
  assert out["max_discrepancy"] == pytest.approx(0.0)


def test_exemption_disabled_fails_an_explained_divergence() -> None:
  ref = _design(uniforms=[0.1, 0.5, 0.9])
  js = _design(sequence=[1, 2, 3, 9], choices=[0, 1, 0], probs=[[0.5, 0.5], [0.5 + 1e-4, 0.5 - 1e-4], [0.5, 0.5]])
  assert gate.compare_designs([js], [ref])["pass"]
  out = gate.compare_designs([js], [ref], exempt_fragile=False)
  assert not out["pass"]
  assert out["fragile"] == 0
  assert [m["field"] for m in out["mismatches"]] == ["sequence"]


def test_exemption_never_excuses_a_plan_field() -> None:
  ref = _design(uniforms=[0.1, 0.5, 0.9])
  js = _design(sequence=[1, 2, 3, 9], choices=[0, 1, 0], pins=[[1, 8, 5]], probs=[[0.5, 0.5], [0.5 + 1e-4, 0.5 - 1e-4], [0.5, 0.5]])
  out = gate.compare_designs([js], [ref])
  assert not out["pass"]
  assert out["fragile"] == 1  # the sequence difference is explained...
  assert [m["field"] for m in out["mismatches"]] == ["pins"]  # ...but the pin mismatch is not


def test_exempt_design_skips_its_energies_and_draws_but_not_the_plan() -> None:
  ref = _design(uniforms=[0.1, 0.5, 0.9], final_potts_energy=-1.0)
  js = _design(sequence=[1, 2, 3, 9], choices=[0, 1, 0], final_potts_energy=5.0, n_draws=99,
               probs=[[0.5, 0.5], [0.5 + 1e-4, 0.5 - 1e-4], [0.5, 0.5]])  # fmt: skip
  out = gate.compare_designs([js], [ref])
  assert out["pass"]
  assert out["fragile"] == 1


def test_without_probs_there_is_no_explanation_and_no_bound() -> None:
  ref = _design(choices=[0, 0, 0], uniforms=[], probs=[])
  js = _design(sequence=[1, 2, 3, 9], choices=[0, 1, 0], probs=[])
  out = gate.compare_designs([js], [ref])
  assert not out["pass"]
  assert out["max_discrepancy"] == 0.0


# ---- cumulative discrepancy and boundary distance ---------------------------------------------------------------------------------------------


def test_cum_discrepancy_of_identical_probs_is_zero() -> None:
  assert gate.cum_discrepancy([0.2, 0.3, 0.5], [0.2, 0.3, 0.5], None) == 0.0


def test_cum_discrepancy_of_a_shifted_distribution_matches_the_hand_value() -> None:
  # cum_js = [0.6, 1.0], cum_ref = [0.5, 1.0]: gaps 0.1 and 0.0
  assert gate.cum_discrepancy([0.6, 0.4], [0.5, 0.5], None) == pytest.approx(0.1)


def test_cum_discrepancy_normalises_each_total() -> None:
  # totals 2 and 4: cum_js / 2 = [0.5, 1.0], cum_ref / 4 = [0.25, 1.0] -> gap 0.25
  assert gate.cum_discrepancy([1.0, 1.0], [1.0, 3.0], None) == pytest.approx(0.25)


def test_boundary_distance_hand_case() -> None:
  # cum = [0.2, 0.5, 1.0], u = 0.45: |cum - 0.45| = [0.25, 0.05, 0.55] -> 0.05 of the total 1
  assert gate.boundary_distance([0.2, 0.3, 0.5], 0.45, None) == pytest.approx(0.05)
  # unnormalised: cum = [1, 2], total 2, u = 0.4 -> target 0.8, |1 - 0.8| = 0.2, relative to 2 -> 0.1
  assert gate.boundary_distance([1.0, 1.0], 0.4, None) == pytest.approx(0.1)


def test_cdf_order_is_honoured() -> None:
  p_js = [0.6, 0.3, 0.1]
  p_ref = [0.1, 0.6, 0.3]
  # identity: cum_js = [0.6, 0.9, 1.0], cum_ref = [0.1, 0.7, 1.0] -> max gap 0.5
  assert gate.cum_discrepancy(p_js, p_ref, None) == pytest.approx(0.5)
  # order [1, 0, 2]: cum_js = [0.3, 0.9, 1.0], cum_ref = [0.6, 0.7, 1.0] -> max gap 0.3
  assert gate.cum_discrepancy(p_js, p_ref, np.array([1, 0, 2])) == pytest.approx(0.3)
  # the boundary distance of u = 0.65 is 0.05 under both orders (nearest cumulative 0.7 identity, 0.6 and 0.7 under [1, 0, 2])
  assert gate.boundary_distance(p_ref, 0.65, None) == pytest.approx(0.05)
  assert gate.boundary_distance(p_ref, 0.65, np.array([1, 0, 2])) == pytest.approx(0.05)


def test_cum_discrepancy_rejects_different_lengths() -> None:
  with pytest.raises(ValueError, match="differ in length"):
    gate.cum_discrepancy([0.5, 0.5], [0.3, 0.3, 0.4], None)


# ---- ordering and fragility (reporting helpers) -----------------------------------------------------------------------------------------------


def test_first_pick_difference_handles_short_streams() -> None:
  assert gate.first_pick_difference([1, 2, 3], [1, 2, 3]) is None
  assert gate.first_pick_difference([1, 2, 3], [1, 9, 3]) == 1
  assert gate.first_pick_difference([1, 2], [1, 2, 3]) == 2


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


# ---- reference and JS normalisation -----------------------------------------------------------------------------------------------------------


def test_normalize_js_maps_pin_fields_choices_probs_and_uniforms() -> None:
  raw = {
    "sample": 1, "sequence": [1, 2], "designable": [0, 1], "label": "",
    "pins": [{"position": 3, "protIdx": 7, "depIdxs": [4], "resId": 9, "type": "HIS-P"}],
    "n_draws": 2, "final_potts_energy": -1.0, "selective_energy": 0.25,
    "events": [{"uniform": 0.3, "choice": 5, "probs": [0.1, 0.9]}, {"uniform": 0.9, "choice": 6, "probs": [0.4, 0.6]}],
  }  # fmt: skip
  got = gate.normalize_js(raw)
  assert got["pins"] == [[3, 7, 9]]
  assert got["choices"] == [5, 6]
  assert got["probs"] == [[0.1, 0.9], [0.4, 0.6]]
  assert got["uniforms"] == [0.3, 0.9]
  assert got["sample"] == 1


def test_ref_dict_carries_probs_and_uniforms() -> None:
  design = SimpleNamespace(
    sample=0, sequence=np.array([1, 2]), designable=np.array([0, 1]), label="",
    pins=[SimpleNamespace(position=1, prot_idx=7, res_id=5)], n_draws=2,
    final_potts_energy=-1.0, selective_energy=0.25,
  )  # fmt: skip
  events = [_Event(np.array([0.5, 0.5]), 0.25, 0), _Event(np.array([0.2, 0.8]), 0.5, 1)]
  ref = gate._ref_dict(design, events, np.array([0, 1]))
  assert ref["probs"] == [[0.5, 0.5], [0.2, 0.8]]
  assert ref["uniforms"] == [0.25, 0.5]
  assert ref["choices"] == [0, 1]
