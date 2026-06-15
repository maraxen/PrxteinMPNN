"""Test that aminx.types.boundaries is a shim pointing to xtrax symbols."""
from aminx.types.boundaries import AxisBoundary, Fuse, Sink, Tap
from xtrax.stages.boundaries import (
    AxisBoundary as XAxisBoundary,
    Fuse as XFuse,
    Sink as XSink,
    Tap as XTap,
)


def test_shim_identity():
    assert Fuse is XFuse, "Fuse must be the same object from xtrax"
    assert Tap is XTap
    assert Sink is XSink
    assert AxisBoundary is XAxisBoundary
