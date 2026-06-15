"""Boundary protocol re-exports from xtrax.stages.

Definitions live in xtrax.stages.boundaries. This shim exists for backward compatibility.
"""
from xtrax.stages.boundaries import AxisBoundary, Fuse, Sink, Tap  # noqa: F401

__all__ = ["AxisBoundary", "Fuse", "Sink", "Tap"]
