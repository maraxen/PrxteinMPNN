"""Synthetic-input checks of the X10 site gate: the outcome ladder, the exact comparators, the planted controls and the resume key.

Every negative here must be caught: a one-ulp float32 change, a changed token or sequence, a missing field, a planted transform that
does not change anything, and an input change that must not reuse a stamped unit.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.browser_validation import protonpotts_site_gate as gate

SYMBOLS = ["A", "C", "D", "E"]


def _good() -> dict:
  return {
    "artifacts_verified": True, "harness_pages_match_graded": True, "site_built": True, "all_units_complete": True,
    "planted_detected": True, "integrity_refused": True, "environment_ok": True, "all_exact": True, "error": "",
  }


def test_decide_passes_an_all_good_record() -> None:
  assert gate._decide(_good()) == "pass"


@pytest.mark.parametrize("key", ["artifacts_verified", "harness_pages_match_graded", "site_built", "all_units_complete"])
def test_decide_error_when_a_prerequisite_fails(key: str) -> None:
  result = _good() | {key: False}
  assert gate._decide(result) == "error"


def test_decide_error_when_error_string_set() -> None:
  assert gate._decide(_good() | {"error": "boom"}) == "error"


@pytest.mark.parametrize("key", ["planted_detected", "integrity_refused", "environment_ok"])
def test_decide_instrument_unverified_when_a_control_is_silent(key: str) -> None:
  assert gate._decide(_good() | {key: False}) == "instrument_unverified"


def test_decide_fail_when_a_cell_differs() -> None:
  assert gate._decide(_good() | {"all_exact": False}) == "fail"


def test_first_match_order_error_beats_instrument_and_fail() -> None:
  broken = _good() | {"all_units_complete": False, "planted_detected": False, "all_exact": False}
  assert gate._decide(broken) == "error"
  assert gate._decide(_good() | {"planted_detected": False, "all_exact": False}) == "instrument_unverified"


def test_scoring_comparator_identical_is_exact() -> None:
  energies = np.array([1.5, -2.25, 3.0], dtype=np.float32)
  tokens = [[1, 2, 3], [4, 5, 6]]
  assert gate.compare_scoring(energies, tokens, energies.tolist(), tokens) == {"exact": True, "mismatches": []}


def test_scoring_comparator_one_ulp_names_energies() -> None:
  energies = np.array([1.5, -2.25, 3.0], dtype=np.float32)
  bumped = energies.copy()
  bumped[1] = np.nextafter(bumped[1], np.float32(np.inf))
  out = gate.compare_scoring(energies, [[1]], bumped.tolist(), [[1]])
  assert out["exact"] is False
  assert out["mismatches"] == ["energies"]


def test_scoring_comparator_changed_token_is_not_exact() -> None:
  energies = np.array([1.0], dtype=np.float32)
  out = gate.compare_scoring(energies, [[1, 2]], energies.tolist(), [[1, 3]])
  assert out["mismatches"] == ["tokens"]


def _design() -> dict:
  return {"sequence": [1, 2, 3], "sample": 0, "method": "block_descent", "label": "hpg", "designable": [True, False, True],
          "pins": [{"position": 0, "protIdx": 4, "depIdxs": [1], "resId": 7, "type": "hpg"}],
          "final_potts_energy": -12.5, "selective_energy": -3.25, "selective_energies": [-1.0, -2.25], "n_draws": 40}


def test_design_comparator_identical_is_exact() -> None:
  designs = [_design()]
  assert gate.compare_design(designs, copy.deepcopy(designs)) == {"exact": True, "mismatches": []}


def test_design_comparator_one_ulp_energy_names_the_field() -> None:
  got = [_design()]
  got[0]["final_potts_energy"] = float(np.nextafter(np.float32(-12.5), np.float32(0.0)))
  out = gate.compare_design([_design()], got)
  assert out["exact"] is False
  assert out["mismatches"] == ["designs[0].final_potts_energy"]


def test_design_comparator_missing_field_is_not_exact() -> None:
  got = [_design()]
  del got[0]["n_draws"]
  assert gate.compare_design([_design()], got)["exact"] is False


def test_sample_comparator_identical_is_exact() -> None:
  ref = {"sequence": [1, 2], "decoding_order": [0, 1], "log_probs": [-0.5, -1.25]}
  assert gate.compare_sample(ref, copy.deepcopy(ref)) == {"exact": True, "mismatches": []}


def test_sample_comparator_one_ulp_log_prob_names_the_field() -> None:
  ref = {"sequence": [1, 2], "decoding_order": [0, 1], "log_probs": [-0.5, -1.25]}
  got = copy.deepcopy(ref)
  got["log_probs"][1] = float(np.nextafter(np.float32(-1.25), np.float32(-10.0)))
  out = gate.compare_sample(ref, got)
  assert out == {"exact": False, "mismatches": ["log_probs"]}


def test_sample_comparator_changed_sequence_is_not_exact() -> None:
  ref = {"sequence": [1, 2], "decoding_order": [0, 1], "log_probs": [-0.5, -1.25]}
  got = copy.deepcopy(ref) | {"sequence": [1, 3]}
  assert gate.compare_sample(ref, got)["mismatches"] == ["sequence"]


def test_sample_comparator_missing_field_is_not_exact() -> None:
  ref = {"sequence": [1, 2], "decoding_order": [0, 1], "log_probs": [-0.5, -1.25]}
  got = {"sequence": [1, 2], "log_probs": [-0.5, -1.25]}
  assert "decoding_order" in gate.compare_sample(ref, got)["mismatches"]


def test_planted_scoring_variants_really_change() -> None:
  variants = [["A", "C"], ["E", "D"]]
  shifted = gate.shift_variant_names(variants, SYMBOLS)
  assert shifted == [["C", "D"], ["A", "E"]]
  assert all(a != b for ra, rb in zip(variants, shifted, strict=True) for a, b in zip(ra, rb, strict=True))


def test_roll_uniforms_changes_the_stream() -> None:
  uniforms = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  rolled = gate.roll_uniforms(uniforms)
  assert rolled == [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]]
  assert rolled != uniforms


def test_roll_flat_changes_a_single_stream() -> None:
  """Sampler cells carry ONE flat stream; roll_uniforms on it was a no-op (run 40a3e57a, sampler control silent)."""
  stream = [1.0, 2.0, 3.0]
  assert gate.roll_flat(stream) == [3.0, 1.0, 2.0]
  assert gate.roll_uniforms(stream) == stream  # the trap: per-element roll of scalars changes nothing


def test_planted_sampler_cell_really_changes_its_stream() -> None:
  cells = [{"id": "1EL1__bias_temp", "uniforms": [0.1, 0.2, 0.3], "noise": [0.0, 0.0, 0.0]}]
  planted = gate.planted_cell("sampler", cells, ["A"])
  assert planted is not None and planted["uniforms"] != cells[0]["uniforms"]
  assert cells[0]["uniforms"] == [0.1, 0.2, 0.3]


def test_planted_cell_is_suffixed_and_perturbed() -> None:
  cells = [{"id": "x__hpg_t0", "bucket": 128, "pdb": "data/x.pdb", "labels": None, "config": {}, "uniforms": None},
           {"id": "x__h_t005", "bucket": 128, "pdb": "data/x.pdb", "labels": None, "config": {}, "uniforms": [[1.0, 2.0]]}]
  planted = gate.planted_cell("ph", cells, SYMBOLS)
  assert planted is not None
  assert planted["id"] == "x__h_t005" + gate.PLANTED
  assert planted["planted_of"] == "x__h_t005"
  assert planted["uniforms"] == [[2.0, 1.0]]
  assert gate.planted_cell("sampler", [{"id": "y__s21", "uniforms": None}], SYMBOLS) is None


def test_message_translation_keeps_bucket_labels_and_uniforms() -> None:
  ph_cell = {"id": "x__h", "bucket": 256, "pdb": "data/x.pdb", "labels": None, "config": {"temperature": 0.05},
             "uniforms": [[0.25]]}
  msg = gate.translate("ph", ph_cell, "pdbs/ph_x.pdb")
  assert msg["kind"] == "design"
  assert msg["params"]["bucket"] == 256
  assert msg["params"]["labels"] is None
  assert msg["params"]["uniforms"] == [[0.25]]
  sample_cell = {"id": "x__s21", "bucket": 128, "pdb": "data/x.pdb", "temperature": 0.1, "seed": 21, "bias": None,
                 "designed": None, "uniforms": None, "noise": None}
  msg_s = gate.translate("sampler", sample_cell, "pdbs/sampler_x.pdb")
  assert msg_s["kind"] == "sample"
  assert msg_s["params"]["seed"] == 21
  assert "uniforms" not in msg_s["params"]
  assert "designed" not in msg_s["params"]
  score = gate.translate("scoring", {"id": "x__main", "bucket": 128, "pdb": "data/x.pdb", "labels": [None],
                                     "variants": [["A"]]}, "pdbs/scoring_x.pdb")
  assert score["kind"] == "score"
  assert score["params"] == {"labels": [None], "variants": [["A"]], "bucket": 128}


def test_inputs_sha_changes_when_any_input_changes() -> None:
  base = {"unit": "deployed_ph_pages", "cells_sha256": "a" * 64, "site_tree_sha256": "b" * 64, "threads": 1,
          "isolate": False, "manifests": {"scoring": "c" * 64}}
  key = gate.inputs_sha256(base)
  assert gate.inputs_sha256(dict(base)) == key
  for field, value in (("cells_sha256", "d" * 64), ("site_tree_sha256", "e" * 64), ("threads", 4), ("isolate", True),
                       ("manifests", {"scoring": "f" * 64})):
    assert gate.inputs_sha256(base | {field: value}) != key, field


def test_flip_byte_changes_exactly_one_byte() -> None:
  data = bytes(range(200))
  flipped = gate.flip_byte(data, 100)
  diffs = [i for i, (a, b) in enumerate(zip(data, flipped, strict=True)) if a != b]
  assert diffs == [100]
  assert flipped[100] == data[100] ^ 0xFF
