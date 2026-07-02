"""Regression tests for SafeMap tile-size resolution (bug #2895).

aminx.tiling.planner's SafeMap demotion used to compute tile size from
tile_granularity alone, ignoring default_batch_size entirely -- contradicting
both AxisSpec.default_batch_size ("positive = safe_map tile size") and
AxisSpec.tile_granularity ("safe_map tile sizes are rounded up to multiples
of this") docstrings. This was masked in production because
src/aminx/tiling/axes.py happens to set tile_granularity == default_batch_size
everywhere except N_RESIDUES/N_LIGAND_ATOMS (which use default_batch_size=0,
a pure-vmap declaration with no safe_map size hint).
"""
from __future__ import annotations

from aminx.tiling.planner import AxisSpec, BatchPlanner, ceil_to_granularity, resolve_safe_map_tile
from aminx.tiling.strategy import SafeMap


def test_ceil_to_granularity_rounds_up() -> None:
    assert ceil_to_granularity(10, 8) == 16
    assert ceil_to_granularity(16, 8) == 16
    assert ceil_to_granularity(1, 1) == 1


def test_ceil_to_granularity_no_rounding_when_granularity_trivial() -> None:
    assert ceil_to_granularity(10, 1) == 10
    assert ceil_to_granularity(10, 0) == 10


def test_resolve_safe_map_tile_uses_default_batch_size_rounded_up() -> None:
    """The core bug: a positive default_batch_size must drive the tile size,
    rounded up to tile_granularity -- not be ignored in favor of
    tile_granularity alone."""
    assert resolve_safe_map_tile(default_batch_size=16, tile_granularity=8) == 16
    assert resolve_safe_map_tile(default_batch_size=10, tile_granularity=8) == 16
    assert resolve_safe_map_tile(default_batch_size=1, tile_granularity=1) == 1


def test_resolve_safe_map_tile_falls_back_to_granularity_when_no_batch_size_declared() -> None:
    """default_batch_size=0 means 'vmap intent, no safe_map size declared'
    (e.g. N_RESIDUES/N_LIGAND_ATOMS). If forced to demote anyway, the only
    positive sizing hint available is tile_granularity -- preserves existing
    behavior for these axes, no regression."""
    assert resolve_safe_map_tile(default_batch_size=0, tile_granularity=128) == 128
    assert resolve_safe_map_tile(default_batch_size=0, tile_granularity=1) == 1


def _planner_forcing_demotion(spec: AxisSpec) -> BatchPlanner:
    return BatchPlanner(
        axes=[spec],
        budget_bytes=0,  # force Phase 2 demotion unconditionally
        estimate_memory=lambda decisions: 1.0,  # any positive estimate exceeds budget_bytes=0
    )


def test_homogeneous_axis_demotion_uses_default_batch_size_not_granularity() -> None:
    """End-to-end planner test: a homogeneous axis with default_batch_size=16,
    tile_granularity=8 (a divergent, previously-bug-triggering configuration)
    must demote to SafeMap(tile=16), not SafeMap(tile=8)."""
    spec = AxisSpec(
        name="batch",
        axis_index=0,
        cardinality=1000,
        default_batch_size=16,
        tile_granularity=8,
        heterogeneous=False,
        doc="regression test axis",
    )
    decision = _planner_forcing_demotion(spec).plan().decisions[0]
    assert isinstance(decision.strategy, SafeMap)
    assert decision.strategy.tile == 16
    assert decision.batch_size == 16


def test_heterogeneous_axis_demotion_uses_default_batch_size_not_granularity() -> None:
    """Same fix applies to Phase 1 (heterogeneous axes, always SafeMap)."""
    spec = AxisSpec(
        name="state",
        axis_index=0,
        cardinality=64,
        default_batch_size=16,
        tile_granularity=8,
        heterogeneous=True,
        doc="regression test axis",
    )
    decision = BatchPlanner(
        axes=[spec],
        budget_bytes=1e12,
        estimate_memory=lambda decisions: 1.0,
    ).plan().decisions[0]
    assert isinstance(decision.strategy, SafeMap)
    assert decision.strategy.tile == 16


def test_default_batch_size_zero_axis_preserves_tile_granularity_fallback() -> None:
    """N_RESIDUES-shaped axis (default_batch_size=0, tile_granularity=128):
    forced demotion must still use 128, not 1 -- no behavior regression for
    the axes that currently rely on the old code path's accidental
    tile_granularity fallback."""
    spec = AxisSpec(
        name="n_residues",
        axis_index=0,
        cardinality=1200,
        default_batch_size=0,
        tile_granularity=128,
        heterogeneous=False,
        doc="N_RESIDUES-shaped regression test axis",
    )
    decision = _planner_forcing_demotion(spec).plan().decisions[0]
    assert isinstance(decision.strategy, SafeMap)
    assert decision.strategy.tile == 128
