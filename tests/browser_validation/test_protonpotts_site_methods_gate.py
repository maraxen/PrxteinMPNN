"""Synthetic-input checks of the X12 site methods gate: the exact comparator, control aggregation, the outcome ladder and the job helpers.

Every negative must be caught: a one-ulp float32 change, a changed int / bool / list element, a missing field, an unaccounted harness field,
a planted transform that changes nothing, and a control that did not fire at one thread setting. No browser, no jax.
"""

from __future__ import annotations

import copy

import numpy as np

from scripts.browser_validation import protonpotts_site_methods_gate as gate


def _design() -> dict:
  return {
    "sequence": [1, 2, 3], "sample": 0, "method": "converged_mcmc", "label": "HIS-P", "designable": [True, False, True],
    "pins": [{"position": 2, "protIdx": 7, "depIdxs": [3], "resId": 41, "type": "HIS-P"}],
    "final_potts_energy": -12.5, "selective_energy": -3.25, "selective_energies": [-1.5, -1.75], "n_draws": 11,
    # harness-only (not returned by the site) and run-description fields
    "events": [{"uniform": 0.5, "choice": 3, "probs": [0.1, 0.9]}], "argmins": [1], "replay_exhausted": False, "visit_order": [2, 0],
  }


def _site(design: dict) -> dict:
  out = {k: copy.deepcopy(v) for k, v in design.items() if k in gate.SITE_COMPARED_FIELDS}
  out["symbols"] = "ACD"  # site-only derived key
  return out


def _ulp(x: float) -> float:
  return float(np.nextafter(np.float32(x), np.float32(np.inf)))


def test_identical_designs_are_exact_and_extra_site_fields_are_ignored() -> None:
  ref = [_design()]
  assert gate.compare_designs(ref, [_site(ref[0])]) == {"exact": True, "mismatches": []}


def test_one_ulp_float32_energy_change_is_caught() -> None:
  ref = [_design()]
  got = _site(ref[0])
  got["final_potts_energy"] = _ulp(got["final_potts_energy"])
  assert got["final_potts_energy"] != ref[0]["final_potts_energy"]
  out = gate.compare_designs(ref, [got])
  assert not out["exact"]
  assert out["mismatches"] == ["designs[0].final_potts_energy"]


def test_one_ulp_inside_a_float_list_is_caught() -> None:
  ref = [_design()]
  got = _site(ref[0])
  got["selective_energies"][1] = _ulp(got["selective_energies"][1])
  assert gate.compare_designs(ref, [got])["mismatches"] == ["designs[0].selective_energies"]


def test_sub_float32_double_difference_is_not_a_float32_mismatch() -> None:
  # float32 BIT equality: two doubles that cast to the same float32 are equal (the harness values are float32 widened to double).
  ref = [_design()]
  got = _site(ref[0])
  got["final_potts_energy"] = ref[0]["final_potts_energy"] + 1e-12
  assert gate.compare_designs(ref, [got])["exact"]


def test_none_float_fields_equal_only_none() -> None:
  ref = [_design() | {"selective_energy": None, "selective_energies": None}]
  assert gate.compare_designs(ref, [_site(ref[0])])["exact"]
  got = _site(ref[0])
  got["selective_energy"] = 0.0
  assert gate.compare_designs(ref, [got])["mismatches"] == ["designs[0].selective_energy"]


def test_int_bool_list_and_pin_changes_are_caught() -> None:
  ref = [_design()]
  for field, value in (("sequence", [1, 2, 4]), ("n_draws", 12), ("designable", [True, True, True]), ("sample", 1), ("label", "ASP-P"),
                       ("method", "gibbs")):
    got = _site(ref[0])
    got[field] = value
    assert gate.compare_designs(ref, [got])["mismatches"] == [f"designs[0].{field}"], field
  got = _site(ref[0])
  got["pins"][0]["resId"] = 42
  assert gate.compare_designs(ref, [got])["mismatches"] == ["designs[0].pins"]


def test_bool_is_not_int_and_int_is_not_float() -> None:
  assert not gate._exact_equal(True, 1)
  assert not gate._exact_equal(1, 1.0)
  assert gate._exact_equal([1, {"a": [2]}], [1, {"a": [2]}])
  assert not gate._exact_equal({"a": 1}, {"a": 1, "b": 2})


def test_run_description_fields_are_ignored_not_compared() -> None:
  ref = [_design()]
  got = _site(ref[0]) | {"wall_ms": 99.0, "user_agent": "x", "chromiumVersion": "1"}
  assert gate.compare_designs(ref, [got])["exact"]


def test_missing_field_and_unaccounted_harness_field_and_count_are_caught() -> None:
  ref = [_design()]
  got = _site(ref[0])
  del got["pins"]
  assert gate.compare_designs(ref, [got])["mismatches"] == ["designs[0].pins:missing"]
  ref2 = [_design() | {"brand_new_field": 1}]
  assert gate.compare_designs(ref2, [_site(ref2[0])])["mismatches"] == ["designs[0].brand_new_field:unaccounted_harness_field"]
  assert not gate.compare_designs(ref, [_site(ref[0]), _site(ref[0])])["exact"]
  assert not gate.compare_designs([], [])["exact"]
  assert not gate.compare_designs(None, [_site(ref[0])])["exact"]


def test_excluded_fields_names_what_is_not_compared() -> None:
  for f in gate.NOT_RETURNED_BY_SITE:
    assert f"NOT_RETURNED_BY_SITE:{f}" in gate.EXCLUDED_FIELDS
  assert "RUN_DESCRIPTION:wall_ms" in gate.EXCLUDED_FIELDS


def test_control_flags_needs_every_key_and_every_row_to_mismatch() -> None:
  keys = [("methods", "pages"), ("methods", "coi")]
  fired = [{"key": k, "exact": False} for k in keys]
  assert gate.control_flags(fired, keys)
  assert not gate.control_flags(fired[:1], keys)  # one thread setting missing
  assert not gate.control_flags([*fired, {"key": keys[0], "exact": True}], keys)  # one row did not fire
  assert not gate.control_flags([], keys)
  assert not gate.control_flags(fired, [])  # nothing expected is not a detection


def _good() -> dict:
  return {"artifacts_verified": True, "harness_pages_match_graded": True, "site_built": True, "all_units_complete": True,
          "planted_detected": True, "order_detected": True, "integrity_refused": True, "environment_ok": True, "all_exact": True, "error": ""}


def test_decide_ladder() -> None:
  assert gate._decide(_good()) == "pass"
  assert gate._decide(_good() | {"all_exact": False}) == "fail"
  for key in ("planted_detected", "order_detected", "integrity_refused", "environment_ok"):
    assert gate._decide(_good() | {key: False}) == "instrument_unverified"
    assert gate._decide(_good() | {key: False, "all_exact": False}) == "instrument_unverified"
  for key in ("artifacts_verified", "harness_pages_match_graded", "site_built", "all_units_complete"):
    assert gate._decide(_good() | {key: False}) == "error"
  assert gate._decide(_good() | {"error": "boom"}) == "error"


def test_blob_problems() -> None:
  assert gate.blob_problems("c", {"a": "1", "b": "2"}, {"a": "1", "b": "2"}) == []
  assert len(gate.blob_problems("c", {"a": "1", "b": "2"}, {"a": "1", "b": "3"})) == 1
  assert len(gate.blob_problems("c", {"a": None}, {"a": "1"})) == 1


def _methods_job(temperature: float, uniforms: list[float]) -> dict:
  return {"path": "methods", "cell": "1EL1", "case": "c", "id": "1EL1__c", "temperature": temperature, "bucket": 256, "labels": None,
          "config": {"method": "gibbs"}, "replay": [{"uniforms": uniforms, "randints": [], "perms": []}] * 2}


def _sweep_job(temperature: float, uniforms: list | None) -> dict:
  return {"path": "sweep", "cell": "1EL1", "case": "knn_t005", "id": "1EL1__knn_t005", "temperature": temperature, "bucket": 256,
          "labels": None, "config": {"sweepOrder": "knn"}, "uniforms": uniforms}


def test_planted_job_changes_the_input_and_refuses_no_ops() -> None:
  job = _methods_job(0.05, [0.1, 0.2, 0.3])
  planted = gate.planted_job(job)
  assert planted is not None
  assert planted["replay"][0]["uniforms"] == [0.3, 0.1, 0.2]
  assert planted["id"].endswith(gate.PLANTED) and planted["planted_of"] == job["id"]
  assert job["replay"][0]["uniforms"] == [0.1, 0.2, 0.3]  # the original is untouched
  assert gate.planted_job(_methods_job(0.05, [0.5, 0.5, 0.5])) is None  # a roll of a constant list changes nothing
  assert gate.planted_job(_methods_job(0.05, [])) is None
  sweep = gate.planted_job(_sweep_job(0.05, [[0.1, 0.2], [0.3, 0.4]]))
  assert sweep is not None and sweep["uniforms"][0][0] == np.float32(0.2).item()
  assert gate.planted_job(_sweep_job(0.0, None)) is None


def test_pick_planted_skips_t0_and_noops() -> None:
  jobs = [_methods_job(0.0, [0.1, 0.2]), _methods_job(0.05, [0.4, 0.4]) | {"id": "x"}, _methods_job(0.05, [0.1, 0.2]) | {"id": "y"}]
  picked = gate.pick_planted(jobs)
  assert picked is not None and picked["planted_of"] == "y"
  assert gate.pick_planted([_methods_job(0.0, [0.1, 0.2])]) is None


def test_order_job_forces_position_and_keeps_uniforms() -> None:
  job = _sweep_job(0.05, [[0.1, 0.2]])
  o = gate.order_job(job)
  assert o["config"]["sweepOrder"] == "position" and o["uniforms"] == job["uniforms"] and o["order_of"] == job["id"]
  assert job["config"]["sweepOrder"] == "knn"
  assert gate.pick_order_job([_sweep_job(0.0, None), _sweep_job(0.05, [[0.1]]) | {"id": "t"}])["id"] == "t"


def test_translate_forwards_full_config_replay_and_uniforms() -> None:
  m = gate.translate(_methods_job(0.05, [0.1, 0.2]), "pdbs/1EL1.pdb")
  assert m["kind"] == "design" and m["params"]["config"] == {"method": "gibbs"} and m["params"]["replay"][0]["uniforms"] == [0.1, 0.2]
  assert "uniforms" not in m["params"]
  s = gate.translate(_sweep_job(0.05, [[0.1]]), "pdbs/1EL1.pdb")
  assert s["params"]["uniforms"] == [[0.1]] and "replay" not in s["params"]
  assert s["params"]["config"]["sweepOrder"] == "knn" and s["params"]["bucket"] == 256


def test_harness_cell_matches_page_job_shape() -> None:
  assert set(gate.harness_cell(_methods_job(0.05, [0.1, 0.2]))) == {"id", "bucket", "labels", "config", "replay"}
  assert set(gate.harness_cell(_sweep_job(0.05, [[0.1]]))) == {"id", "bucket", "labels", "config", "uniforms"}


def test_sweep_uniforms_seeded_and_cell_specific() -> None:
  a = gate.sweep_uniforms(5031, 0, 0.05)
  assert a == gate.sweep_uniforms(5031, 0, 0.05)
  assert a != gate.sweep_uniforms(4031, 0, 0.05) and a != gate.sweep_uniforms(5031, 1, 0.05)
  assert gate.sweep_uniforms(5031, 0, 0.0) is None
