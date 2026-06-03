"""Tests for Phase 0 carry pre-demotion in BatchPlanner.plan()."""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from prxteinmpnn.tiling.axes import N_NOISES, N_SAMPLES, N_STRUCTURES, N_TEMPERATURES
from prxteinmpnn.tiling.carry import CarrySpec
from prxteinmpnn.tiling.planner import AxisDecision, AxisSpec, BatchPlanner
from prxteinmpnn.tiling.strategy import SafeMap, Scan, Vmap


def _make_planner(axes: list[AxisSpec], carries: list[CarrySpec] = None) -> BatchPlanner:
    """Helper to construct a planner with optional carries."""
    if carries is None:
        carries = []
    return BatchPlanner(
        axes=list(axes),
        budget_bytes=1e12,  # huge budget so Phase 2 never demotes
        estimate_memory=lambda ds: 1.0,
        carries=list(carries),
    )


def test_axis_decision_has_strategy_field() -> None:
    """AxisDecision should have a strategy field."""
    d = AxisDecision(
        axis=N_NOISES,
        batch_size=0,
        reasoning="vmap",
        strategy=Vmap(),
    )
    assert isinstance(d.strategy, Vmap)


def test_axis_decision_batch_size_zero_with_vmap_strategy() -> None:
    """AxisDecision with batch_size=0 should have Vmap strategy."""
    d = AxisDecision(axis=N_NOISES, batch_size=0, reasoning="vmap", strategy=Vmap())
    assert d.batch_size == 0
    assert isinstance(d.strategy, Vmap)


def test_planner_phase0_demotes_carry_axis_to_scan() -> None:
    """Phase 0 should demote carry-bearing axes to Scan strategy."""
    init = jnp.zeros(16)
    carry = CarrySpec(
        axis_name="n_noises",
        init=init,
        transition=lambda c, x: (c + x, c),
    )
    planner = _make_planner([N_NOISES, N_SAMPLES], carries=[carry])
    plan = planner.plan()
    noise_decision = plan.decision_for("n_noises")
    assert isinstance(noise_decision.strategy, Scan)
    assert noise_decision.strategy.init is init


def test_planner_phase0_does_not_affect_non_carry_axes() -> None:
    """Phase 0 should not affect axes without CarrySpec."""
    carry = CarrySpec(
        axis_name="n_noises",
        init=None,
        transition=lambda c, x: (c, x),
    )
    planner = _make_planner([N_NOISES, N_SAMPLES], carries=[carry])
    plan = planner.plan()
    sample_decision = plan.decision_for("n_samples")
    # With huge budget, homogeneous axis should remain vmap
    assert isinstance(sample_decision.strategy, Vmap)


def test_planner_phase1_heterogeneous_axes_still_safe_map() -> None:
    """Phase 1 should assign SafeMap to heterogeneous axes."""
    planner = _make_planner([N_STRUCTURES, N_NOISES])
    plan = planner.plan()
    struct_decision = plan.decision_for("n_structures")
    assert isinstance(struct_decision.strategy, SafeMap)
    assert struct_decision.strategy.tile >= 1


def test_planner_phase0_carry_on_heterogeneous_axis_raises() -> None:
    """CarrySpec should reject heterogeneous axis names in __post_init__."""
    # N_STRUCTURES is heterogeneous, so this should raise
    with pytest.raises(ValueError, match="heterogeneous"):
        CarrySpec(
            axis_name="n_structures",
            init=None,
            transition=lambda c, x: (c, x),
        )


def test_planner_treedef_invariant_strategies_are_static() -> None:
    """AxisStrategy values on Vmap/SafeMap should have no JAX array leaves."""
    import jax

    planner = _make_planner([N_NOISES, N_SAMPLES])
    plan = planner.plan()
    for d in plan.decisions:
        # tree_leaves includes the dataclass itself as a leaf if it has no JAX array fields
        # We check that there are no actual JAX arrays inside
        leaves = jax.tree_util.tree_leaves(d.strategy)
        if isinstance(d.strategy, (Vmap, SafeMap)):
            # For Vmap, tree_leaves returns [Vmap()] (the dataclass itself)
            # For SafeMap, tree_leaves returns [SafeMap(...)] (the dataclass itself)
            # This is expected; we're asserting no JAX arrays are inside
            for leaf in leaves:
                assert not isinstance(
                    leaf, (jnp.ndarray, type(jnp.array(1)))
                ), f"Strategy {d.strategy} contains JAX array: {leaf}"


def test_planner_phase0_carry_init_is_preserved() -> None:
    """The Scan strategy should preserve the exact init object."""
    init_arr = jnp.array([1.0, 2.0, 3.0])
    carry = CarrySpec(
        axis_name="n_temperatures",
        init=init_arr,
        transition=lambda c, x: (c + x, c),
    )
    planner = _make_planner([N_TEMPERATURES], carries=[carry])
    plan = planner.plan()
    temp_decision = plan.decision_for("n_temperatures")
    assert isinstance(temp_decision.strategy, Scan)
    # Verify init is the same object (identity, not just equality)
    assert temp_decision.strategy.init is init_arr


def test_dedup_spec_rejected_on_non_eligible_axis() -> None:
    """BatchPlanner.plan() should raise TilingError if DedupGather is assigned to
    an axis with dedup_eligible=False."""
    from prxteinmpnn.tiling.dedup import DedupSpec
    from prxteinmpnn.tiling.errors import TilingError
    import numpy as np

    # N_NOISES has dedup_eligible=False (default); assigning DedupGather should fail
    dedup_spec = DedupSpec(
        axis_name="n_noises",
        unique_indices=np.array([0], dtype=np.int32),
        index_map=np.array([0, 0], dtype=np.int32),
        k=1,
    )
    planner = _make_planner([N_NOISES], dedup_specs=[dedup_spec])
    with pytest.raises(TilingError, match="dedup_eligible"):
        planner.plan()
