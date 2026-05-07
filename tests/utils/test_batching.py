import pytest
from prxteinmpnn.utils.batching import (
    AxisDecision,
    AxisSpec,
    BatchPlan,
    BatchPlanner,
    ceil_to_granularity,
    estimate_memory_theoretical,
)


def _axis(name, index, cardinality, default_bs, tile_gran, heterogeneous=False):
    return AxisSpec(
        name=name,
        axis_index=index,
        cardinality=cardinality,
        default_batch_size=default_bs,
        tile_granularity=tile_gran,
        heterogeneous=heterogeneous,
        doc="",
    )


# --- ceil_to_granularity ---

def test_ceil_already_aligned():
    assert ceil_to_granularity(128, 128) == 128

def test_ceil_unaligned():
    assert ceil_to_granularity(129, 128) == 256

def test_ceil_granularity_one():
    assert ceil_to_granularity(7, 1) == 7

def test_ceil_zero():
    assert ceil_to_granularity(0, 128) == 0


# --- estimate_memory_theoretical ---

def test_estimate_all_vmap():
    ax = _axis("a", 0, 8, 0, 1)
    d = AxisDecision(axis=ax, batch_size=0, reasoning="vmap")
    assert estimate_memory_theoretical([d], 1.0, 1.0) == 8.0

def test_estimate_safe_map_tile_1():
    ax = _axis("a", 0, 8, 1, 1)
    d = AxisDecision(axis=ax, batch_size=1, reasoning="safe_map")
    assert estimate_memory_theoretical([d], 1.0, 1.0) == 1.0

def test_estimate_safe_map_tile_equals_cardinality_same_as_vmap():
    ax = _axis("a", 0, 8, 8, 1)
    d_vmap = AxisDecision(axis=ax, batch_size=0, reasoning="vmap")
    d_safe = AxisDecision(axis=ax, batch_size=8, reasoning="safe_map cardinality")
    assert estimate_memory_theoretical([d_vmap], 1.0, 1.0) == \
           estimate_memory_theoretical([d_safe], 1.0, 1.0)

def test_estimate_mixed_two_axes():
    ax_states = _axis("n_states", 0, 4, 0, 1)
    ax_temps = _axis("n_temperatures", 1, 4, 1, 1)
    decisions = [
        AxisDecision(axis=ax_states, batch_size=0, reasoning="vmap"),
        AxisDecision(axis=ax_temps, batch_size=1, reasoning="safe_map"),
    ]
    assert estimate_memory_theoretical(decisions, 1.0, 2.5) == pytest.approx(10.0)

def test_estimate_multiplier_applied():
    ax = _axis("a", 0, 4, 0, 1)
    d = AxisDecision(axis=ax, batch_size=0, reasoning="vmap")
    assert estimate_memory_theoretical([d], 2.0, 3.0) == pytest.approx(24.0)


# --- BatchPlanner.plan() ---

def _planner(axes, budget):
    return BatchPlanner(
        axes=axes,
        budget_bytes=budget,
        estimate_memory=lambda ds: estimate_memory_theoretical(ds, 1.0, 1.0),
    )


def test_planner_no_demotion_needed():
    ax = _axis("n_samples", 0, 8, 0, 1)
    plan = _planner([ax], budget=1000.0).plan()
    assert plan.decision_for("n_samples").batch_size == 0
    assert not plan.exceeded_budget()

def test_planner_demotes_innermost_first():
    ax0 = _axis("n_states", 0, 4, 0, 1)
    ax1 = _axis("n_samples", 1, 8, 0, 1)
    # budget=8: vmap both → 4×8=32 > 8; demote ax0 (innermost) → 1×8=8 <= 8
    plan = _planner([ax0, ax1], budget=8.0).plan()
    assert plan.decision_for("n_states").batch_size == 1
    assert plan.decision_for("n_samples").batch_size == 0

def test_planner_tile_granularity_respected():
    ax = _axis("n_residues", 0, 1200, 0, 128)
    # budget=64: vmap → 1200 > 64; demote → tile=128
    plan = _planner([ax], budget=64.0).plan()
    assert plan.decision_for("n_residues").batch_size == 128

def test_planner_heterogeneous_always_safe_map_ignores_budget():
    ax = _axis("n_structures", 0, 32, 0, 1, heterogeneous=True)
    plan = _planner([ax], budget=1e9).plan()
    assert plan.decision_for("n_structures").batch_size == 1

def test_planner_exceeded_budget_flag():
    ax = _axis("n_residues", 0, 1200, 0, 128)
    # budget=0.5: even tile=128 → 128 > 0.5
    plan = _planner([ax], budget=0.5).plan()
    assert plan.exceeded_budget()

def test_planner_exceeded_budget_returns_bool_never_raises():
    ax = _axis("n_residues", 0, 1200, 0, 128)
    plan = _planner([ax], budget=0.5).plan()
    result = plan.exceeded_budget()
    assert isinstance(result, bool)

def test_plan_decision_for_unknown_raises():
    ax = _axis("n_samples", 0, 8, 0, 1)
    plan = _planner([ax], budget=1000.0).plan()
    with pytest.raises(KeyError):
        plan.decision_for("nonexistent")

def test_planner_multi_axis_cascade_demotion():
    """Innermost axes are demoted before outer ones when budget is tight."""
    ax0 = _axis("n_states", 0, 4, 0, 1)   # axis_index=0 (innermost)
    ax1 = _axis("n_samples", 1, 8, 0, 1)  # axis_index=1
    ax2 = _axis("n_temps", 2, 4, 0, 1)    # axis_index=2 (outermost)
    # budget=16: all vmap → 4×8×4=128 > 16
    # demote ax0: 1×8×4=32 > 16
    # demote ax1: 1×1×4=4 <= 16 → stops here; ax2 stays vmap
    plan = _planner([ax0, ax1, ax2], budget=16.0).plan()
    assert plan.decision_for("n_states").batch_size == 1   # demoted
    assert plan.decision_for("n_samples").batch_size == 1  # demoted
    assert plan.decision_for("n_temps").batch_size == 0    # vmap (budget met after ax1 demoted)
