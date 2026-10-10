"""Tests for the block-descent visit orders (``sweep_order`` knn / energy), pure numpy on tiny tables.

Upstream reference: ``potts_mpnn_ph.py`` ``_knn_rank`` (2180-2190), ``_centre_energy_rank`` (2141-2177).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aminx.families.protonpotts_mpnn.ph_config import PHDesignConfig
from aminx.families.protonpotts_mpnn.ph_descent import _check_supported
from aminx.families.protonpotts_mpnn.ph_design import _centre_energy_rank, _visit_order
from aminx.families.protonpotts_mpnn.ph_plan import Pin, Plan


def _pin(position: int, prot: int, deps: tuple[int, ...]) -> Pin:
  return Pin(position=position, protonation_type="HIS-P", prot_idx=prot, dep_idxs=deps, res_id=position)


def test_knn_rank_orders_by_best_rank_then_index() -> None:
  # Centre 0 lists [3, 2] as its kNN row; centre 4 lists [2, 1]. Best ranks: 2 -> 0 (from centre 4), 3 -> 0,
  # 1 -> 1. Tie between 2 and 3 at rank 0 breaks on index.
  e_idx = np.array(
    [[0, 3, 2], [1, 0, 2], [2, 0, 1], [3, 0, 1], [4, 2, 1], [5, 0, 1], [6, 0, 1]],
    dtype=np.int32,
  )
  plan = Plan(pins=(_pin(0, 0, (1,)), _pin(4, 0, (1,))), designable=(1, 2, 3), label="t")
  visit = _visit_order(PHDesignConfig(sweep_order="knn"), plan, np.zeros((7, 3, 2, 2)), e_idx, np.zeros(7))
  assert tuple(visit) == (2, 3, 1)


def test_knn_rank_unranked_position_sorts_last_by_index() -> None:
  # Position 6 appears in no centre's kNN row, so it is unranked and sorts after every ranked position.
  # Position 5 is also unranked and comes before 6 by index.
  e_idx = np.array(
    [[0, 3, 2], [1, 0, 2], [2, 0, 1], [3, 0, 1], [4, 2, 1], [5, 0, 1], [6, 0, 1]],
    dtype=np.int32,
  )
  plan = Plan(pins=(_pin(0, 0, (1,)), _pin(4, 0, (1,))), designable=(1, 2, 3, 5, 6), label="t")
  visit = _visit_order(PHDesignConfig(sweep_order="knn"), plan, np.zeros((7, 3, 2, 2)), e_idx, np.zeros(7))
  assert tuple(visit) == (2, 3, 1, 5, 6)


def test_energy_rank_uses_combined_out_and_in_terms_and_orders_by_score() -> None:
  # L=3, K=3, V=2. Centre 0 (pin: prot token 0, deps (1,)). Designable 1 and 2.
  # Seed sequence: seq[1]=0, seq[2]=1.
  # Position 1: out-edge 0->1 gives e_p=3, e_d=0; in-edge 1->0 gives e_p=0, e_d=3. The two terms are summed
  #   before the abs (upstream), so score = |3 - 3| = 0. A sum-of-abs implementation would give 6.
  # Position 2: out-edge 0->2 gives e_p=1.5, e_d=0.5; in-edge 2->0 gives e_p=0.75, e_d=0.25.
  #   score = |2.25 - 0.75| = 1.5. Position 2 is more important, so it comes first.
  e_idx = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]], dtype=np.int32)
  table = np.zeros((3, 3, 2, 2))
  table[0, 1, 0, 0] = 3.0   # out 0->1, [a_0=prot, a_1=seq[1]=0]
  table[0, 1, 1, 0] = 0.0   # out 0->1, [a_0=dep,  a_1=0]
  table[1, 1, 0, 0] = 0.0   # in 1->0, [a_1=0, a_0=prot]
  table[1, 1, 0, 1] = 3.0   # in 1->0, [a_1=0, a_0=dep]
  table[0, 2, 0, 1] = 1.5   # out 0->2, [a_0=prot, a_2=seq[2]=1]
  table[0, 2, 1, 1] = 0.5   # out 0->2, [a_0=dep,  a_2=1]
  table[2, 1, 1, 0] = 0.75  # in 2->0, [a_2=1, a_0=prot]
  table[2, 1, 1, 1] = 0.25  # in 2->0, [a_2=1, a_0=dep]
  seq = np.array([0, 0, 1], dtype=np.int32)
  order = _centre_energy_rank(table, e_idx, [_pin(0, 0, (1,))], seq, [1, 2])
  assert order == [2, 1]


def test_energy_rank_ties_break_on_ascending_position() -> None:
  # An all-zero table gives every position score 0, so the order is ascending index.
  e_idx = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]], dtype=np.int32)
  order = _centre_energy_rank(np.zeros((3, 3, 2, 2)), e_idx, [_pin(0, 0, (1,))], np.zeros(3, np.int32), [2, 1])
  assert order == [1, 2]


def test_position_order_is_unchanged() -> None:
  e_idx = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]], dtype=np.int32)
  plan = Plan(pins=(_pin(0, 0, (1,)),), designable=(2, 1), label="t")
  visit = _visit_order(PHDesignConfig(sweep_order="position"), plan, np.zeros((3, 3, 2, 2)), e_idx, np.zeros(3))
  assert tuple(visit) == (2, 1)


@pytest.mark.parametrize(("knob", "value"), [
  ("global_weight", 0.1),
  ("adjacent_repeat_weight", 0.2),
  ("self_weight", 0.9),
  ("zscale_mode", "single_mutation"),
])
def test_other_unsupported_knobs_still_refused(knob: str, value: object) -> None:
  with pytest.raises(NotImplementedError, match=knob):
    _check_supported(PHDesignConfig(**{knob: value}))


@pytest.mark.parametrize("order", ["position", "knn", "energy"])
def test_ported_sweep_orders_are_accepted(order: str) -> None:
  _check_supported(PHDesignConfig(sweep_order=order))


def test_unknown_sweep_order_raises() -> None:
  # Rejected at construction by PHDesignConfig.__post_init__.
  with pytest.raises(ValueError, match="sweep_order"):
    PHDesignConfig(sweep_order="bogus")
  # And by the descent guard if a config somehow carries it.
  bad = SimpleNamespace(
    global_weight=0.0,
    adjacent_repeat_weight=0.0,
    self_weight=1.0,
    zscale_mode="block",
    sweep_order="bogus",
  )
  with pytest.raises(NotImplementedError, match="sweep_order"):
    _check_supported(bad)  # type: ignore[arg-type]

