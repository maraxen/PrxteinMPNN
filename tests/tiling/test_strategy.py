# tests/tiling/test_strategy.py
from __future__ import annotations
import jax.numpy as jnp
import pytest
from prxteinmpnn.tiling.strategy import Vmap, SafeMap, Scan, AxisStrategy


def test_vmap_is_frozen_dataclass():
    v = Vmap()
    assert v == Vmap()  # equality by value


def test_safe_map_stores_tile():
    s = SafeMap(tile=4)
    assert s.tile == 4
    assert s == SafeMap(tile=4)
    assert s != SafeMap(tile=8)


def test_scan_stores_init_and_transition():
    init = jnp.zeros((5, 32))
    def transition(carry, x):
        return carry + x, carry
    s = Scan(init=init, transition=transition)
    assert s.ordered_sinks is True  # default


def test_scan_ordered_sinks_configurable():
    s = Scan(init=None, transition=lambda c, x: (c, x), ordered_sinks=False)
    assert s.ordered_sinks is False


def test_axis_strategy_union_isinstance():
    assert isinstance(Vmap(), (Vmap, SafeMap, Scan))
    assert isinstance(SafeMap(tile=2), (Vmap, SafeMap, Scan))
    assert isinstance(Scan(init=None, transition=lambda c, x: (c, x)), (Vmap, SafeMap, Scan))


def test_scan_transition_protocol_conformance():
    from prxteinmpnn.tiling.strategy import ScanTransition
    # A function with (carry, x) -> (carry, y) signature satisfies the protocol
    def my_transition(carry: int, x: int) -> tuple[int, int]:
        return carry + x, carry
    assert isinstance(my_transition, ScanTransition)
