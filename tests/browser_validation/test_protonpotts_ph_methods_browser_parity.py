"""Pure aggregation checks of the X11c gate: control detection across thread settings, isolated_at_4, unit counting. No browser, no jax."""

from __future__ import annotations

from scripts.browser_validation import protonpotts_ph_methods_browser_parity as gate


def _controls(**flagged: bool) -> dict:
  c = gate.new_controls()
  for name, value in flagged.items():
    c[name]["flagged"] = value
  return c


def test_control_detected_only_if_fired_at_every_thread_setting() -> None:
  all_on = dict.fromkeys(gate.CONTROL_NAMES, True)
  by_t = {1: _controls(**all_on), 4: _controls(**(all_on | {"perm": False}))}
  got = gate.aggregate_controls(by_t, (1, 4))
  assert got["control_tokens_detected"] is True
  assert got["control_perm_detected"] is False  # fired at 1 thread only


def test_control_not_detected_when_a_thread_setting_is_missing_or_empty() -> None:
  all_on = dict.fromkeys(gate.CONTROL_NAMES, True)
  assert not any(gate.aggregate_controls({1: _controls(**all_on)}, (1, 4)).values())
  assert not any(gate.aggregate_controls({}, (1, 4)).values())


def test_isolated_at_4_requires_isolation_and_more_than_one_thread() -> None:
  ok = {"threads": 4, "cross_origin_isolated": True, "num_threads_effective": 4}
  assert gate.isolated_at_4([{"threads": 1, "cross_origin_isolated": False, "num_threads_effective": 1}, ok]) is True
  assert gate.isolated_at_4([ok | {"num_threads_effective": 1}]) is False  # fell back to one thread
  assert gate.isolated_at_4([ok | {"cross_origin_isolated": False}]) is False
  assert gate.isolated_at_4([ok, ok | {"cross_origin_isolated": False}]) is False  # every 4-thread unit must qualify
  assert gate.isolated_at_4([{"threads": 1, "cross_origin_isolated": True, "num_threads_effective": 1}]) is False  # no 4-thread unit


def test_threads_effective_is_the_minimum_per_setting() -> None:
  units = [
    {"threads": 1, "num_threads_effective": 1},
    {"threads": 4, "num_threads_effective": 4},
    {"threads": 4, "num_threads_effective": 1},
  ]
  assert gate.threads_effective(units) == {"1": 1, "4": 1}


def test_count_units_spans_thread_settings() -> None:
  rows = [{"threads": t, "passed": p} for t, p in ((1, True), (1, True), (4, True), (4, False))]
  assert gate.count_units(rows) == (4, 3)
  assert gate.count_units([]) == (0, 0)
