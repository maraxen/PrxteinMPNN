"""Parity suite for make_axis_dispatch_via_xtrax vs make_axis_dispatch.

EPIC #1541 / T2.4 (`.praxia/docs/specs/260611_aminx-xtrax-refactor.md`).
Mirrors tests/tiling/test_dispatch.py's assertions exactly, but against the
xtrax-backed `make_axis_dispatch_via_xtrax`, plus direct side-by-side checks.
This is the shadow-parity evidence for flipping `factory.py` (and friends)
from `make_axis_dispatch` to `make_axis_dispatch_via_xtrax` once T2.GATE
passes -- if this suite passes, the xtrax-backed path is behaviorally
indistinguishable from the current one for every case the original suite
covers.
"""

from __future__ import annotations

import numpy as np
import pytest

from aminx.host.plan import PlanTopologyError
from aminx.tiling.dispatch import (
    DispatchRejected,
    make_axis_dispatch,
    make_axis_dispatch_via_xtrax,
)
from aminx.tiling.errors import TilingError
from aminx.tiling.strategy import DedupGather, SafeMap, Scan, Vmap


class TestMakeAxisDispatchViaXtraxHappyPath:
    """Mirrors TestMakeAxisDispatchHappyPath in test_dispatch.py."""

    def test_vmap_dispatch_returns_vmap_iterator(self) -> None:
        result = make_axis_dispatch_via_xtrax(Vmap())
        assert result is not None
        assert type(result).__name__ == "VmapIterator"

    def test_safemap_dispatch_returns_safemap_iterator(self) -> None:
        result = make_axis_dispatch_via_xtrax(SafeMap(tile=4))
        assert result is not None
        assert type(result).__name__ == "SafeMapIterator"
        assert result.tile == 4


class TestMakeAxisDispatchViaXtraxRejectPath:
    """Mirrors TestMakeAxisDispatchRejectPath in test_dispatch.py."""

    def test_scan_on_state_axis_raises_dispatch_rejected(self) -> None:
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))

        with pytest.raises(DispatchRejected) as exc_info:
            make_axis_dispatch_via_xtrax(scan_strategy, axis="state")

        error_msg = str(exc_info.value)
        assert "state axis" in error_msg.lower()
        assert "heterogeneous" in error_msg.lower()

    def test_dispatch_rejected_is_still_a_tilingerror(self) -> None:
        """The translated exception (aminx's own DispatchRejected) preserves
        the TilingError contract -- callers doing `except TilingError` keep
        working even though xtrax's own DispatchRejected does not subclass it.
        """
        assert issubclass(DispatchRejected, TilingError)
        assert issubclass(PlanTopologyError, TilingError)

    def test_make_axis_dispatch_via_xtrax_rejects_dedup_gather(self) -> None:
        dg = DedupGather(
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1, 0, 1], dtype=np.int32),
            k=2,
            k_bucket=2,
        )
        with pytest.raises(DispatchRejected):
            make_axis_dispatch_via_xtrax(dg)


class TestSideBySideParity:
    """Direct comparison: same input, same observable outcome, both paths."""

    def test_vmap_produces_same_iterator_type_on_both_paths(self) -> None:
        legacy = make_axis_dispatch(Vmap())
        via_xtrax = make_axis_dispatch_via_xtrax(Vmap())
        assert type(legacy).__name__ == type(via_xtrax).__name__ == "VmapIterator"

    def test_safemap_produces_same_tile_value_on_both_paths(self) -> None:
        legacy = make_axis_dispatch(SafeMap(tile=7))
        via_xtrax = make_axis_dispatch_via_xtrax(SafeMap(tile=7))
        assert legacy.tile == via_xtrax.tile == 7

    def test_scan_on_state_rejected_on_both_paths(self) -> None:
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))
        with pytest.raises(DispatchRejected):
            make_axis_dispatch(scan_strategy, axis="state")
        with pytest.raises(DispatchRejected):
            make_axis_dispatch_via_xtrax(scan_strategy, axis="state")

    def test_scan_on_non_state_axis_accepted_on_both_paths(self) -> None:
        """Confirms the heterogeneous_axes translation covers the non-rejection
        case too, not just the rejection case -- Scan on a non-heterogeneous
        axis must be accepted on both paths, not just "doesn't crash."
        """
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))
        legacy = make_axis_dispatch(scan_strategy, axis="wave")
        via_xtrax = make_axis_dispatch_via_xtrax(scan_strategy, axis="wave")
        assert type(legacy).__name__ == type(via_xtrax).__name__ == "JaxScanIterator"

    def test_dedup_gather_rejected_on_both_paths(self) -> None:
        dg = DedupGather(
            unique_indices=np.array([0, 1], dtype=np.int32),
            index_map=np.array([0, 1, 0, 1], dtype=np.int32),
            k=2,
            k_bucket=2,
        )
        with pytest.raises(DispatchRejected):
            make_axis_dispatch(dg)
        with pytest.raises(DispatchRejected):
            make_axis_dispatch_via_xtrax(dg)
