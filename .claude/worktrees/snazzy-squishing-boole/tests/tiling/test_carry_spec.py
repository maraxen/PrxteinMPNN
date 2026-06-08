"""Tests for CarrySpec — carry declaration for planner Phase 0."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from prxteinmpnn.tiling.carry import CarrySpec
from prxteinmpnn.tiling.strategy import ScanTransition


def test_carry_spec_stores_axis_name():
    def t(carry, x): return carry, x
    cs = CarrySpec(axis_name="n_noises", init=jnp.zeros(32), transition=t)
    assert cs.axis_name == "n_noises"


def test_carry_spec_stores_init_and_transition():
    init = jnp.ones((5, 32))
    def t(carry, x): return carry + x, carry
    cs = CarrySpec(axis_name="n_samples", init=init, transition=t)
    assert cs.init.shape == (5, 32)
    assert cs.transition is t


def test_carry_spec_default_ordered_sinks():
    cs = CarrySpec(axis_name="n_noises", init=None, transition=lambda c, x: (c, x))
    assert cs.ordered_sinks is True


def test_carry_spec_ordered_sinks_configurable():
    cs = CarrySpec(
        axis_name="n_noises", init=None,
        transition=lambda c, x: (c, x), ordered_sinks=False
    )
    assert cs.ordered_sinks is False


def test_carry_spec_rejects_heterogeneous_axis_name():
    """CarrySpec should not be created for axes known to be heterogeneous."""
    with pytest.raises(ValueError, match="heterogeneous"):
        CarrySpec(
            axis_name="n_structures",
            init=jnp.zeros(32),
            transition=lambda c, x: (c, x),
        )


def test_carry_spec_rejects_other_known_heterogeneous():
    with pytest.raises(ValueError, match="heterogeneous"):
        CarrySpec(
            axis_name="n_states",
            init=jnp.zeros(32),
            transition=lambda c, x: (c, x),
        )
