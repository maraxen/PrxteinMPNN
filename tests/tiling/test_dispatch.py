"""Test suite for make_axis_dispatch library-side factory contract (Task S6-A2).

This module verifies the dispatch logic that converts AxisStrategy instances
into typed iterators, with rejection logic for invalid axis/strategy pairs.
"""

from __future__ import annotations

import pytest

from prxteinmpnn.host.plan import PlanTopologyError
from prxteinmpnn.tiling.dispatch import DispatchRejected, make_axis_dispatch
from prxteinmpnn.tiling.strategy import SafeMap, Scan, Vmap


class TestMakeAxisDispatchHappyPath:
    """Happy path: valid strategy/axis pairs dispatch to correct iterator."""

    def test_vmap_dispatch_returns_vmap_iterator(self) -> None:
        """make_axis_dispatch(Vmap()) returns a VmapIterator."""
        result = make_axis_dispatch(Vmap())
        assert result is not None
        # Verify it's the right type by checking the class name.
        assert type(result).__name__ == "VmapIterator"

    def test_safemap_dispatch_returns_safemap_iterator(self) -> None:
        """make_axis_dispatch(SafeMap(tile=4)) returns a SafeMapIterator with tile=4."""
        result = make_axis_dispatch(SafeMap(tile=4))
        assert result is not None
        assert type(result).__name__ == "SafeMapIterator"
        # SafeMapIterator has a static tile field.
        assert result.tile == 4


class TestMakeAxisDispatchRejectPath:
    """Reject path: Scan on heterogeneous (state) axis raises DispatchRejected."""

    def test_scan_on_state_axis_raises_dispatch_rejected(self) -> None:
        """make_axis_dispatch(Scan(...), axis="state") raises DispatchRejected."""
        # Create a minimal Scan strategy.
        scan_strategy = Scan(init=0, transition=lambda c, x: (c, x))

        with pytest.raises(DispatchRejected) as exc_info:
            make_axis_dispatch(scan_strategy, axis="state")

        error_msg = str(exc_info.value)
        # Error message must mention both "state axis" and "heterogeneous".
        assert "state axis" in error_msg.lower()
        assert "heterogeneous" in error_msg.lower()

    def test_dispatch_rejected_is_plan_topology_error(self) -> None:
        """DispatchRejected is a subclass of PlanTopologyError."""
        assert issubclass(DispatchRejected, PlanTopologyError)
