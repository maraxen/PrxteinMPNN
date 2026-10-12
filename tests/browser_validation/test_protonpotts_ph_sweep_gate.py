"""Pure aggregation and comparison checks of the X7d gate: control detection per setting, visit-order grading, first-difference location. No browser, no jax."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.browser_validation import protonpotts_ph_sweep_gate as gate

SETTINGS = ("node@1", "chromium@1", "chromium@4")


def _controls(**counts: tuple[int, int]) -> dict:
  c = gate.new_controls()
  for name, (applies, flagged) in counts.items():
    c[name] = {"applies": applies, "flagged": flagged}
  return c


def _all_on() -> dict:
  return _controls(order=(3, 3), tokens=(2, 2), energy=(1, 1))


def test_control_detected_only_if_fired_on_every_case_at_every_setting() -> None:
  by = dict.fromkeys(SETTINGS, _all_on())
  assert all(gate.aggregate_controls(by, SETTINGS).values())
  by["chromium@4"] = _controls(order=(3, 2), tokens=(2, 2), energy=(1, 1))  # one applicable case not flagged at one setting
  got = gate.aggregate_controls(by, SETTINGS)
  assert got["control_order_detected"] is False
  assert got["control_tokens_detected"] is True


def test_control_not_detected_when_it_never_applied_or_a_setting_is_missing() -> None:
  by = dict.fromkeys(SETTINGS, _controls(order=(0, 0), tokens=(2, 2), energy=(1, 1)))
  assert gate.aggregate_controls(by, SETTINGS)["control_order_detected"] is False  # knob never live: not evidence
  assert not any(gate.aggregate_controls({"node@1": _all_on()}, SETTINGS).values())
  assert not any(gate.aggregate_controls({}, SETTINGS).values())


def test_first_diff_index() -> None:
  assert gate.first_diff_index([1, 2, 3], [1, 2, 3]) is None
  assert gate.first_diff_index([1, 2, 3], [1, 9, 3]) == 1
  assert gate.first_diff_index([1, 2], [1, 2, 3]) == 2


def _ref(seq: list[int]) -> SimpleNamespace:
  pin = SimpleNamespace(position=3, prot_idx=7, res_id=30)
  return SimpleNamespace(sequence=np.asarray(seq), designable=(1, 2, 3), pins=(pin,), label="L", n_draws=0,
                         final_potts_energy=-5.0, selective_energy=1.0)


def _js(visit: list[int] | None, seq: list[int]) -> dict:
  return {"sample": 0, "sequence": seq, "designable": [1, 2, 3], "pins": [{"position": 3, "protIdx": 7, "resId": 30}], "label": "L",
          "n_draws": 0, "final_potts_energy": -5.0, "selective_energy": 1.0, "visit_order": visit}


def test_grade_unit_passes_only_with_equal_visit_order() -> None:
  ref = [_ref([0, 1, 2])]
  ok = gate.grade_unit([_js([2, 1, 3], [0, 1, 2])], ref, [2, 1, 3])
  assert ok["passed"] is True
  assert ok["first_difference"] is None
  bad = gate.grade_unit([_js([1, 2, 3], [0, 1, 2])], ref, [2, 1, 3])  # sequences equal, order differs: still a failure
  assert bad["passed"] is False
  assert bad["first_difference"] == {"sample": 0, "field": "visit_order", "index": 0}
  missing = gate.grade_unit([_js(None, [0, 1, 2])], ref, [2, 1, 3])  # a port that records no order cannot pass
  assert missing["passed"] is False
  assert missing["first_difference"]["field"] == "visit_order"


def test_locate_difference_reports_first_sequence_index_when_order_matches() -> None:
  ref = [_ref([0, 1, 2, 4])]
  got = gate.locate_difference([_js([2, 1, 3], [0, 1, 9, 4])], ref, [2, 1, 3])
  assert got == {"sample": 0, "field": "sequence", "index": 2}
  assert gate.locate_difference([], ref, [2, 1, 3])["field"] == "n_designs"


def test_uniforms_are_per_cell_temperature_and_reproducible() -> None:
  assert gate.uniforms_for(0, 0.0) is None
  a, b, c = gate.uniforms_for(1, 0.05), gate.uniforms_for(1, 0.05), gate.uniforms_for(2, 0.05)
  assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
  assert not np.array_equal(a[0], c[0])
  assert len(a) == gate.SAMPLES
