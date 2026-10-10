"""Synthetic-input checks of the X9 bench's validity predicates: positive controls AND inputs that must fail.

The bench's only live control is a planted delay that can only pass; these feed `_complete`, `_summarize` and `_decide` constructed payloads
(an empty, truncated, short-rep or errored cell; a wrong thread count; no isolation; no warm-up) and require `incomplete` / `timer_blind`.
"""

from __future__ import annotations

import copy

import pytest

from scripts.browser_validation import protonpotts_browser_bench as bench

REPS = 3


def _sampler_cell(length: int = 10, reps: int = REPS) -> dict:
  return {"id": "c", "ok": True, "wall_ms_all": [10.0] * reps, "warmup_ms_all": [50.0],
          "rep_sig_all": [[length, length * 30]] * reps, "sequence": [1] * length, "log_probs": [0.0] * (length * 30)}


SAMPLER_EXPECT = [10, 300]


def test_complete_accepts_a_full_cell() -> None:
  assert bench._complete("sampler", _sampler_cell(), REPS, SAMPLER_EXPECT)


@pytest.mark.parametrize(
  "mutate",
  [
    lambda c: c.update(ok=False),
    lambda c: c.update(wall_ms_all=[10.0] * (REPS - 1)),
    lambda c: c.update(rep_sig_all=[[10, 300]] * (REPS - 1)),
    lambda c: c.update(rep_sig_all=[[10, 300], [10, 300], [9, 270]]),  # one truncated repetition
    lambda c: c.update(rep_sig_all=[[0, 0]] * REPS),  # fast-failed empty result
    lambda c: c.update(sequence=[]),
  ],
)
def test_complete_rejects_bad_cells(mutate) -> None:  # noqa: ANN001
  cell = copy.deepcopy(_sampler_cell())
  mutate(cell)
  assert not bench._complete("sampler", cell, REPS, SAMPLER_EXPECT)


def test_none_expectation_needs_positive_value() -> None:
  assert bench._sig_matches([20, 1.5], [20, None])
  assert not bench._sig_matches([20, 0], [20, None])


def _payload(threads_effective: int = 4, isolated: bool = True, measured: float = 250.0) -> dict:
  return {"cells": [_sampler_cell()], "planted_ms": 250.0, "ctrl_measured_ms": measured, "num_threads_effective": threads_effective,
          "cross_origin_isolated": isolated, "warmup": 1, "session_create_ms": {"256": 1.0}}


def _summ(payload: dict, requested: int = 4) -> dict:
  run: dict = {"threads_requested": requested}
  bench._summarize("sampler", run, payload, {"c": SAMPLER_EXPECT}, REPS)
  return run


def test_summarize_clean_run() -> None:
  run = _summ(_payload())
  assert run["isolation_ok"] and run["ctrl_in_window"] and run["cells_complete"] == 1
  assert run["per_cell_warmup_ms"] == {"c": [50.0]}


def test_isolation_fails_on_thread_fallback_or_no_coop() -> None:
  assert not _summ(_payload(threads_effective=1))["isolation_ok"]  # asked for 4, ORT fell back to 1
  assert not _summ(_payload(isolated=False))["isolation_ok"]


def test_timer_control_window() -> None:
  assert not _summ(_payload(measured=0.0))["ctrl_in_window"]  # a clock that did not see the planted delay
  assert not _summ(_payload(measured=100.0))["ctrl_in_window"]  # coarsened/too short
  assert not _summ(_payload(measured=250.0 + 2500.0))["ctrl_in_window"]


def _result(**over: object) -> dict:
  base = {"bench_ok": True, "arms_total": 3, "cells_total": 16, "reps": 3, "warmup": 1, "isolation_ok": True, "shape_ok": True,
          "git_clean": True, "failure": "", "ctrl_in_window": True}
  base.update(over)
  return base


def test_decide_pass_and_each_negative() -> None:
  assert bench._decide(_result()) == "pass"
  for key, val in [("bench_ok", False), ("arms_total", 2), ("cells_total", 5), ("reps", 2), ("warmup", 0),
                   ("isolation_ok", False), ("shape_ok", False), ("git_clean", False), ("failure", "boom")]:
    assert bench._decide(_result(**{key: val})) == "incomplete", key
  assert bench._decide(_result(ctrl_in_window=False)) == "timer_blind"
  assert bench._decide(_result(ctrl_in_window=False, shape_ok=False)) == "incomplete"  # first match wins
